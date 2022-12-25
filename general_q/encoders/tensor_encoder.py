from typing import Generic, TypeVar

from abc import ABC, abstractmethod

import numpy as np
import torch
from gymnasium import spaces

from general_q.encoders.base import Batch, Encoder, I


class TensorBatch(Batch[torch.Tensor, I], Generic[I]):
    @property
    def data(self):
        return self._data

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value.data


M = TypeVar("M", bound=torch.nn.Module)


class TensorEncoder(Encoder[TensorBatch, I], ABC, Generic[M, I]):
    def __init__(self, space: spaces.Space[I], embed_dim: int):
        super().__init__(space)
        self.dtype = torch.tensor(space.sample()).dtype
        self.encoder: M = self.make_encoder(space, embed_dim)

    @abstractmethod
    def make_encoder(self, space: spaces.Space[I], embed_dim: int) -> M:
        pass

    def prepare(self, sample: I) -> TensorBatch:
        return TensorBatch(
            torch.tensor(sample, dtype=self.dtype, device=self.device),
            self,
        )

    def sample(self, batch_size=1, minibatch_size=64) -> TensorBatch:
        # TODO experiment with different minibatch sizes
        samples: list[torch.Tensor] = []
        for i in range(0, batch_size, minibatch_size):
            minibatch = np.array([
                self.space.sample()
                for _ in range(min(batch_size - i, minibatch_size))
            ])

            samples.append(self.prepare(minibatch).data)

        return TensorBatch(torch.cat(samples), self)

    def forward(self, x: TensorBatch) -> torch.FloatTensor:
        return self.encoder(x.data)

    def item(self, x):
        return np.array(x.data.cpu())
