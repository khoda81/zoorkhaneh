from typing import Callable, Generic, TypeVar

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
from gymnasium import spaces
from torch import nn

T = TypeVar("T", bound="Batched")


class BatchProxy:
    def __init__(self, batch):
        self.batch = batch

    def __getitem__(self, item):
        return self.batch.batched_getitem(item)

    def __setitem__(self, item, value):
        self.batch.batched_setitem(item, value)


class Batched(Generic[T]):
    def __init__(self, data):
        self.data = data

    @property
    def batch(self):
        return BatchProxy(self)

    def batched_getitem(self, item) -> "T":
        return self.__class__(self.data[item])

    def batched_setitem(self, item, value: "T"):
        self.data[item] = value.data

    def apply(self, func: Callable[["Batched"], "Batched"]) -> "Batched":
        return self.__class__(func(self.data))

    def __repr__(self) -> str:
        return self.data.__repr__()


B = TypeVar("B", bound=Batched)
I = TypeVar("I")


class Encoder(nn.Module, ABC, Generic[I, B]):
    def __init__(self, space: spaces.Space[I], *args, **kwargs) -> None:
        assert self.supports(space), f"{self.__class__.__name__} does not support {space}"
        super().__init__(*args, **kwargs)
        self.space = space
        # phantom data to track device
        self.register_buffer("_phantom", torch.empty((0,)))

    __call__: Callable[["Encoder", B], torch.FloatTensor]

    @property
    def device(self) -> torch.device:
        return self._phantom.device

    @staticmethod
    @abstractmethod
    def supports(space: spaces.Space) -> bool:
        """Returns whether this encoder supports the given space."""
        return False

    @abstractmethod
    def prepare(self, sample: I) -> B:
        """Prepare sample for forward pass."""

    @abstractmethod
    def sample(self, batch_shape: Iterable[int] = ()) -> B:
        """Sample a batch of samples from the space."""

    @abstractmethod
    def forward(self, sample: B) -> torch.FloatTensor:
        """Encode the given sample."""

    @abstractmethod
    def item(self, sample):
        """Convert a sample to the python type."""
