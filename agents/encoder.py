import math
import torch
import numpy as np
from abc import ABC, abstractclassmethod, abstractmethod
from torch import nn
from gym.spaces import Space, Box, Discrete, MultiDiscrete, Tuple, Dict, MultiBinary


class Encoder(ABC, nn.Module):
    def __init__(self, space: Space):
        assert self.supports(space), f"{self.__class__.__name__} does not support {space}"

    def __init__(self, space: Space):
        super().__init__()
        self.space = space
        # phantom data to track device
        self.register_buffer("_phantom", torch.empty((0,)))

    @property
    def device(self):
        return self._phantom.device

    @abstractclassmethod
    def supports(cls, space: Space) -> bool:
        """
        Returns whether this encoder supports the given space.
        """

    @abstractmethod
    def prepare(self, sample):
        """
        Prepare numpy sample for forward pass.
        """

    @abstractmethod
    def sample(self, batch_size=1):
        """
        Sample a batch of samples from the space.
        """

    @abstractmethod
    def concat(self, sample_batch, new_sample):
        """
        Concat a new sample to the given sample batch.
        """

    @abstractmethod
    def getitem(self, sample_batch, item):
        pass

    @abstractmethod
    def shape(self, sample_batch):
        """
        Return the batch shpae of the given sample batch.
        """

    @abstractmethod
    def forward(self, x):
        """
        Encode the given sample.
        """

    def item(self, x):
        return np.array(x)


class TensorEncoder(Encoder):
    def __init__(self, space: Box, encoder: nn.Module):
        super().__init__(space)
        self.dtype = torch.tensor(space.sample()).dtype
        self.encoder = encoder

    def prepare(self, sample):
        return torch.tensor(sample, dtype=self.dtype, device=self.device)

    def sample(self, batch_size=1, minibatch_size=64):
        sample = torch.empty(batch_size, *self.space.shape, dtype=self.dtype, device=self.device)
        for i in range(0, batch_size, minibatch_size):
            mini_size = len(sample[i:i+minibatch_size])
            minibatch = np.array([
                self.space.sample()
                for _ in range(mini_size)
            ])

            sample[i:i+minibatch_size] = self.prepare(minibatch)

        return sample

    def concat(self, sample_batch, new_sample):
        return torch.cat([sample_batch, new_sample])

    def getitem(self, sample, item):
        return sample[item]

    def shape(self, sample_batch):
        return sample_batch.shape[:-len(self.space.shape)]

    def forward(self, x):
        return self.encoder(x)

    def item(self, x):
        return np.array(x.to(torch.device('cpu')))


class BoxEncoder(TensorEncoder):
    def __init__(self, space: Box, embed_dim=256):
        super().__init__(
            space,
            nn.Sequential(
                nn.Flatten(start_dim=-len(space.shape)),
                nn.Linear(math.prod(space.shape), embed_dim),
            )
        )

    @classmethod
    def supports(cls, space: Space) -> bool:
        return type(space) == Box


class DiscreteEncoder(TensorEncoder):
    def __init__(self, space: Discrete, embed_dim=256):
        super().__init__(space, nn.Embedding(space.n, embed_dim))

    @staticmethod
    def supports(space: Space) -> bool:
        return isinstance(space, Discrete)


class MultiDiscreteEncoder(TensorEncoder):
    def __init__(self, space: MultiDiscrete, embed_dim=256):
        super().__init__(
            space,
            nn.ModuleList([
                DiscreteEncoder(Discrete(n), embed_dim)
                for n in space.nvec
            ])
        )

        self.dtype = torch.float32

    @staticmethod
    def supports(space: Space) -> bool:
        return type(space) == MultiDiscrete

    def forward(self, x):
        return torch.sum(
            torch.stack(
                [encoder(x[..., i]) for i, encoder in enumerate(self.encoders)],
                dim=-1
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
            )
        )

    @staticmethod
    def supports(space: Space) -> bool:
        return type(space) == MultiBinary


class TupleEncoder(Encoder):
    def __init__(self, space: Tuple, embed_dim=256):
        super().__init__(space)
        self.encoders = nn.ModuleList([
            auto_encoder(subspace, embed_dim)
            for subspace in space.spaces
        ])

    @staticmethod
    def supports(space: Space) -> bool:
        return isinstance(space, Tuple)

    def sample(self, batch_size=1):
        return tuple(
            encoder.sample(batch_size)
            for encoder in self.encoders
        )

    def concat(self, sample_batch, new_sample):
        return tuple(
            encoder.concat(subsample_batch, subsample)
            for encoder, subsample_batch, subsample
            in zip(self.encoders, sample_batch, new_sample)
        )

    def getitem(self, sample_batch, item):
        return tuple(
            encoder.getitem(subsample_batch, item)
            for encoder, subsample_batch
            in zip(self.encoders, sample_batch)
        )

    def shape(self, sample_batch):
        for encoder, subsample_batch in zip(self.encoders, sample_batch):
            shape = encoder.shape(subsample_batch)
            if shape:
                return shape

    def prepare(self, sample):
        return tuple(
            encoder.prepare(subsample)
            for encoder, subsample
            in zip(self.encoders, sample)
        )

    def forward(self, x):
        return torch.sum(
            torch.stack(
                [encoder(x[i]) for i, encoder in enumerate(self.encoders)],
                dim=-1
            ),
            dim=-1,
        )

    def item(self, sample):
        return tuple(
            encoder.item(subsample)
            for encoder, subsample
            in zip(self.encoders, sample)
        )


class DictEncoder(Encoder):
    def __init__(self, space: Dict, embed_dim=256):
        super().__init__(space)
        self.encoders = nn.ModuleDict({
            key: auto_encoder(subspace, embed_dim)
            for key, subspace in space.spaces.items()
        })

    @staticmethod
    def supports(space: Space) -> bool:
        return isinstance(space, Dict)

    def concat(self, sample_batch, new_sample):
        return {
            key: encoder.concat(sample_batch[key], new_sample[key])
            for key, encoder
            in self.encoders.items()
        }

    def getitem(self, sample_batch, item):
        return {
            key: encoder.getitem(sample_batch[key], item)
            for key, encoder
            in self.encoders.items()
        }

    def shape(self, sample_batch):
        for key, encoder in self.encoders.items():
            shape = encoder.shape(sample_batch[key])
            if shape:
                return shape

    def prepare(self, sample):
        return {
            key: encoder.prepare(sample[key])
            for key, encoder
            in self.encoders.items()
        }

    def forward(self, x):
        return torch.sum(
            torch.stack(
                [
                    encoder(x[key])
                    for key, encoder
                    in self.encoders.items()
                ],
                dim=-1
            ),
            dim=-1,
        )

    def item(self, sample):
        return {
            key: encoder.item(sample[key])
            for key, encoder
            in self.encoders.items()
        }


encoders = [
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
