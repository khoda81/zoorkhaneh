from typing import Generic, TypeVar

import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from gymnasium import Space, spaces
from numpy.typing import NDArray
from torch import nn

from general_q.encoders.base import Encoder, I
from general_q.encoders.storage import TensorStorage

M = TypeVar("M", bound=torch.nn.Module)


def unflatten(tensor: torch.Tensor, dim, sizes):
    if len(sizes) == 0:
        return tensor.squeeze(dim=dim)
    else:
        return tensor.unflatten(dim=dim, sizes=sizes)


class TensorEncoder(Encoder[I, TensorStorage], ABC, Generic[I]):
    def __init__(self, space: Space[I], *args, **kwargs):
        super().__init__(space)
        sample = torch.tensor(space.sample())
        self.dtype = sample.dtype
        self.sample_shape = sample.shape
        self.encoder = self.make_encoder(*args, **kwargs)

    @abstractmethod
    def make_encoder(self, *args, **kwargs) -> nn.Module:
        pass

    def prepare(self, sample: I) -> TensorStorage:
        sample = torch.tensor(sample, dtype=self.dtype, device=self.device)
        return TensorStorage(sample)

    def unprepare(self, sample: TensorStorage) -> NDArray:
        return sample.data.cpu().numpy()

    def sample(self, batch_shape=(), mini_batch_size=4096) -> TensorStorage:
        sample: list[torch.Tensor] = []
        batch_size = math.prod(batch_shape)
        if mini_batch_size <= 0:
            mini_batch_size = batch_size

        for i in range(0, batch_size, mini_batch_size):
            mini_batch = np.array([
                self.space.sample()
                for _ in range(min(batch_size - i, mini_batch_size))
            ])

            sample.append(self.prepare(mini_batch).data)

        sample = torch.cat(sample)
        sample = unflatten(sample, dim=0, sizes=batch_shape)
        return TensorStorage(sample)

    def forward(self, sample: TensorStorage) -> torch.FloatTensor:
        return self.encoder(sample.data).unsqueeze(-2)


class FloatTensorEncoder(TensorEncoder):
    space: spaces.Box

    @classmethod
    def supports(cls, space: Space) -> bool:
        dtypes = [np.float16, np.float32, np.float64]
        return super().supports(space) and space.dtype in dtypes

    def make_encoder(self, embed_dim):
        return nn.Linear(math.prod(self.space.shape), embed_dim)

    def prepare(self, sample):
        sample = super().prepare(sample)
        batch_len = len(sample.shape) - len(self.space.shape)
        assert sample.shape[batch_len:] == self.space.shape, \
            f"Sample shape={tuple(sample.shape)} does not match expected shape: " \
            f"(*batch_shape, {', '.join(map(str, self.space.shape))})"

        new_shape = sample.shape[:batch_len] + (-1,)
        sample.data = sample.data.reshape(new_shape)
        return sample

    def sample(self, batch_shape=(), *args, **kwargs) -> TensorStorage:
        batch_shape = tuple(batch_shape)
        repeats = batch_shape + (1,) * len(self.space.shape)

        space = spaces.Box(
            low=np.tile(self.space.low,  repeats),
            high=np.tile(self.space.high, repeats),
            shape=batch_shape + self.space.shape,
            dtype=self.space.dtype,
        )

        return self.prepare(space.sample())
