import functools
import math
from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
import torch
from gym.spaces import *
from torch import nn


class Sample:
    def __init__(self, encoder, data):
        self.encoder = encoder
        self.data = data

    @staticmethod
    def wrap_out(func):
        """Decorator to wrap a function that returns a sample."""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            return Sample(self, func(self, *args, **kwargs))

        return wrapper

    @staticmethod
    def unwrap_inp(func):
        """Decorator to wrap a function that returns a sample."""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            args = (
                arg.data if isinstance(arg, Sample) else arg for arg in args
            )

            kwargs = {
                key: value.data if isinstance(value, Sample) else value
                for key, value in kwargs.items()
            }

            return func(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def wrap(func):
        """Decorator to wrap a function that returns a sample."""
        return Sample.wrap_out(Sample.unwrap_inp(func))

    def map(self, func):
        return self.encoder.map(self, func)

    def item(self):
        return self.encoder.item(self)

    def __repr__(self):
        return f"Sample({self.encoder}, {self.data})"

    def __getitem__(self, key):
        return self.map(lambda x: x[key])

    def __setitem__(self, key, value):
        for dst, src in self.encoder.zip(self.data, value.data):
            dst.data[key] = src.data


class Encoder(nn.Module, ABC):
    def __init__(self, space) -> None:
        assert self.supports(
            space
        ), f"{self.__class__.__name__} does not support {space}"

        super().__init__()
        self.space = space
        # phantom data to track device
        self.register_buffer("_phantom", torch.empty((0,)))

    @property
    def device(self) -> torch.device:
        return self._phantom.device

    @classmethod
    def auto_encoder(cls, space: Space, embed_dim=256) -> "Encoder":
        for encoder in [
            cls,
            ByteImageEncoder,
            BoxEncoder,
            DiscreteEncoder,
            MultiDiscreteEncoder,
            MultiBinaryEncoder,
            TupleEncoder,
            DictEncoder,
        ]:
            if encoder.supports(space):
                return encoder(space, embed_dim=embed_dim)

        raise ValueError(f"No encoder found for {space}. Consider writing one.")

    @staticmethod
    def supports(space) -> bool:
        """
        Returns whether this encoder supports the given space.
        """
        return False

    @abstractmethod
    def prepare(self, sample):
        """
        Prepare numpy sample for forward pass.
        """

    @abstractmethod
    def zip(self, *sample_batch):
        """
        Zip the given sample batches.
        """

    @abstractmethod
    def map(self, sample_batch, func):
        """
        Map the given function to the batch.
        """

    @abstractmethod
    def sample(self, batch_size: int = 1):
        """
        Sample a batch of samples from the space.
        """

    @abstractmethod
    def shape(self, sample_batch) -> tuple:
        """
        Return the batch shape of the given sample batch.
        """

    @abstractmethod
    def forward(self, x) -> torch.FloatTensor:
        """
        Encode the given sample.
        """

    def item(self, x):
        return np.array(x)


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
        self.encoders = nn.ModuleDict({
            key: self.auto_encoder(subspace, embed_dim)
            for key, subspace in space.spaces.items()
        })

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
