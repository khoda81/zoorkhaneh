import functools
from abc import ABC, abstractmethod

import numpy as np
import torch
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


class Encoder(ABC, nn.Module):
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

    @staticmethod
    @abstractmethod
    def supports(space) -> bool:
        """
        Returns whether this encoder supports the given space.
        """

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

    def getitem(self, sample, item):
        return self.map(sample, lambda x: x[item])

    def setitem(self, sample, item, value):
        return self.map(sample, lambda x: x.__setitem__(item, value))

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
