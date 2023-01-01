from typing import Callable, Generic, TypeVar

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
from gymnasium import Space
from torch import nn

from general_q.encoders.storage import Storage

B = TypeVar("B", bound=Storage)
I = TypeVar("I")


class UnsupportedSpaceError(Exception):
    pass


class Encoder(nn.Module, ABC, Generic[I, B]):
    def __init__(self, space: Space[I], *args, **kwargs):
        if not self.supports(space):
            raise UnsupportedSpaceError(f"{self.__class__} does not support {space}")

        super().__init__(*args, **kwargs)
        self.space = space
        # phantom data to track device
        self.register_buffer("_phantom", torch.empty((0,)))

    __call__: Callable[["Encoder", B], torch.FloatTensor]

    @property
    def device(self) -> torch.device:
        return self._phantom.device

    @classmethod
    def supports(cls, space):
        try:
            return (
                not hasattr(cls, "__annotations__") or
                "space" not in cls.__annotations__ or
                isinstance(space, cls.__annotations__["space"])
            )
        except TypeError:
            raise TypeError(
                f"Failed to determine if {space} is supported by {cls}. "
                f"Consider writing a `supports` method for the encoder."
            )

    @abstractmethod
    def prepare(self, sample: I) -> B:
        """Prepare sample for forward pass."""

    @abstractmethod
    def unprepare(self, sample):
        """Convert a sample to the python type."""

    @abstractmethod
    def sample(self, batch_shape: Iterable[int] = ()) -> B:
        """
        Sample a batch of samples from the space. Returned sample is always prepared.

        Args:
            batch_shape: Shape of the sample batch.
        """

    # TODO add empty() method or an argument to sample, for empty samples
    # TODO useful when we need a dummy sample and sample is too slow for it

    @abstractmethod
    def forward(self, sample: B) -> torch.FloatTensor:
        """
        Encode the given sample and return encoded tensor.

        Args:
            sample: A prepared sample.

        Returns:
            A batch of sequence of tokens: (*batch_shape, T, embed_dim), where T
            is the number of tokens.
        """
