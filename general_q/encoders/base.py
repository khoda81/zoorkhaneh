from typing import Callable, Generic, TypeVar

from abc import ABC, abstractmethod

import torch
from gym import spaces
from torch import nn

T = TypeVar("T")
I = TypeVar("I")


class Batch(ABC, Generic[T, I]):
    def __init__(self, data: T, encoder) -> None:
        self._data = data
        self.encoder = encoder

    @property
    def data(self):
        return self._data

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __setitem__(self, item, value) -> None:
        pass

    def item(self):
        return self.encoder.item(self)


class Encoder(nn.Module, ABC, Generic[T, I]):
    def __init__(self, space: spaces.Space[I]) -> None:
        assert self.supports(
            space
        ), f"{self.__class__.__name__} does not support {space}"

        super().__init__()
        self.space = space

        # phantom data to track device
        self.register_buffer("_phantom", torch.empty((0,)))

    __call__: Callable[["Encoder", T], torch.FloatTensor]

    @property
    def device(self) -> torch.device:
        return self._phantom.device

    @staticmethod
    def supports(space: spaces.Space) -> bool:
        """Returns whether this encoder supports the given space."""
        return False

    @abstractmethod
    def prepare(self, sample: I) -> T:
        """Prepare sample for forward pass."""

    @abstractmethod
    def sample(self, batch_size: int = 1) -> T:
        """Sample a batch of samples from the space."""

    @abstractmethod
    def forward(self, x: T) -> torch.FloatTensor:
        """Encode the given sample."""

    @abstractmethod
    def item(self, x):
        """Convert a tensor to a sample."""
