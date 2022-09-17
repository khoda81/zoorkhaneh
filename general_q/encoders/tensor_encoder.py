import math
from abc import ABC
from itertools import chain

import numpy as np
import torch
from gym.spaces import *
from torch import nn

from general_q.encoders import Encoder, Sample


class TensorEncoder(Encoder, ABC):
    def __init__(self, space: Space, encoder: nn.Module):
        super().__init__(space)
        self.dtype = torch.tensor(space.sample()).dtype
        self.encoder = encoder

    @Sample.wrap_out
    def prepare(self, sample):
        return torch.tensor(sample, dtype=self.dtype, device=self.device)

    @Sample.unwrap_inp
    def zip(self, *sample_batch):
        yield sample_batch

    @Sample.wrap
    def map(self, sample_batch, func):
        return func(sample_batch)

    @Sample.wrap_out
    def sample(self, batch_size=1, minibatch_size=64):
        # TODO experiment with different minibatch sizes
        samples = []
        for i in range(0, batch_size, minibatch_size):
            minibatch = np.array(
                [
                    self.space.sample()
                    for _ in range(min(batch_size - i, minibatch_size))
                ]
            )

            samples.append(self.prepare(minibatch).data)

        return torch.cat(samples)

    def shape(self, sample_batch):
        return sample_batch.data.shape[: -len(self.space.shape)]

    @Sample.unwrap_inp
    def forward(self, x):
        return self.encoder(x)

    @Sample.unwrap_inp
    def item(self, x):
        return np.array(x.to(torch.device("cpu")))


class ByteImageEncoder(TensorEncoder):
    def __init__(self, space: Box, embed_dim=256):
        super(TensorEncoder, self).__init__(space)
        self.dtype = torch.uint8

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=-3),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample = self.sample(1)
            sample = sample.to(torch.float32) / 255.0 - 0.5
            _, n = self(sample).shape

        self.encoder.append(nn.Linear(n, embed_dim))

    @staticmethod
    def supports(space: Space) -> bool:
        if not (
            type(space) == Box
            and space.dtype == np.uint8
            and len(space.shape) == 3
            and space.low.min() == 0
            and space.high.max() == 255
        ):
            return False

        h, w, c = space.shape
        return c == 3 and h >= 30 and w >= 30

    @Sample.wrap_out
    def prepare(self, sample):
        # rearrange last three dimensions to be (c, h, w)
        return super().prepare(sample).transpose(-1, -3)

    @Sample.unwrap_inp
    def forward(self, x):
        return self.encoder(x.to(torch.float32) / 255.0 - 0.5)

    @Sample.unwrap_inp
    def item(self, x):
        x = x.transpose(-1, -3).to(torch.device("cpu"))
        return np.array(x, dtype=np.uint8)


class BoxEncoder(TensorEncoder):
    def __init__(self, space: Box, embed_dim=256):
        super().__init__(
            space,
            nn.Sequential(
                nn.Flatten(start_dim=-len(space.shape)),
                nn.Linear(math.prod(space.shape), embed_dim),
            ),
        )

    @classmethod
    def supports(cls, space: Space) -> bool:
        return type(space) == Box and space.dtype in [
            np.float16,
            np.float32,
            np.float64,
        ]


class DiscreteEncoder(TensorEncoder):
    def __init__(self, space: Discrete, embed_dim=256):
        super().__init__(space, nn.Embedding(space.n, embed_dim))

    @staticmethod
    def supports(space: Space) -> bool:
        return isinstance(space, Discrete)

    @Sample.unwrap_inp
    def item(self, x):
        return x.item()


class MultiDiscreteEncoder(TensorEncoder):
    def __init__(self, space: MultiDiscrete, embed_dim=256):
        super().__init__(
            space,
            nn.ModuleList(
                [DiscreteEncoder(Discrete(n), embed_dim) for n in space.nvec]
            ),
        )

        self.dtype = torch.float32

    @staticmethod
    def supports(space: Space) -> bool:
        return type(space) == MultiDiscrete

    @Sample.unwrap_inp
    def forward(self, x):
        return torch.sum(
            torch.stack(
                [encoder(x[..., i]) for i, encoder in enumerate(self.encoders)],
                dim=-1,
            ),
            dim=-1,
        )


class MultiBinaryEncoder(TensorEncoder):
    def __init__(self, space: MultiBinary, embed_dim=256):
        super().__init__(
            space,
            nn.Sequential(
                nn.Flatten(start_dim=-len(space.shape)),
                nn.Linear(space.n, embed_dim),
            ),
        )

    @staticmethod
    def supports(space: Space) -> bool:
        return type(space) == MultiBinary


class DictEncoder(Encoder):
    def __init__(self, space: Dict, embed_dim=256):
        super().__init__(space)
        self.encoders = nn.ModuleDict(
            {
                key: auto_encoder(subspace, embed_dim)
                for key, subspace in space.spaces.items()
            }
        )

    @staticmethod
    def supports(space: Space) -> bool:
        return isinstance(space, Dict)

    @Sample.wrap_out
    def prepare(self, sample):
        return {
            key: encoder.prepare(sample[key])
            for key, encoder in self.encoders.items()
        }

    @Sample.unwrap_inp
    def zip(self, *sample_batches):
        return chain.from_iterable(
            [
                encoder.zip(
                    *map(lambda sample_batch: sample_batch[key], sample_batches)
                )
                for key, encoder in self.encoders.items()
            ]
        )

    @Sample.wrap
    def map(self, sample_batch, func):
        return {
            key: encoder.map(sample_batch[key], func)
            for key, encoder in self.encoders.items()
        }

    @Sample.wrap_out
    def sample(self, batch_size=1):
        return {
            key: encoder.sample(batch_size=batch_size)
            for key, encoder in self.encoders.items()
        }

    def shape(self, sample_batch):
        for key, encoder in self.encoders.items():
            shape = encoder.shape(sample_batch[key])
            if shape:
                return shape

    @Sample.unwrap_inp
    def forward(self, x):
        return torch.sum(
            torch.stack(
                [encoder(x[key]) for key, encoder in self.encoders.items()],
                dim=-1,
            ),
            dim=-1,
        )

    @Sample.unwrap_inp
    def item(self, sample):
        return {
            key: encoder.item(sample[key])
            for key, encoder in self.encoders.items()
        }


class TupleEncoder(DictEncoder):
    def __init__(self, space: Tuple, embed_dim=256):
        space = Dict({i: subspace for i, subspace in enumerate(space.spaces)})

        super().__init__(space, embed_dim)

    @staticmethod
    def supports(space: Space) -> bool:
        return isinstance(space, Tuple)

    def prepare(self, sample):
        return super().prepare(dict(enumerate(sample)))

    def item(self, sample):
        return tuple(super().item(sample).values())


encoders = [
    ByteImageEncoder,
    BoxEncoder,
    DiscreteEncoder,
    MultiDiscreteEncoder,
    MultiBinaryEncoder,
    TupleEncoder,
    DictEncoder,
]


def auto_encoder(space: Space, embed_dim=256):
    """
    Automatically select an encoder for a given space.
    """
    for encoder in encoders:
        if encoder.supports(space):
            return encoder(space, embed_dim)

    raise ValueError(f"No encoder found for space {space}")


def register_encoder(encoder: Encoder, index=0):
    """
    Register a new encoder.
    """
    encoders.insert(index, encoder)
