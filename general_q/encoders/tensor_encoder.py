from typing import Generic, TypeVar

import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from gymnasium import spaces

from general_q.encoders.base import Batched, Encoder, I

M = TypeVar("M", bound=torch.nn.Module)


class TensorEncoder(Encoder[I, Batched], ABC, Generic[I, M]):
    def __init__(self, space: spaces.Space[I], *args, **kwargs):
        super().__init__(space)
        self.dtype = torch.tensor(space.sample()).dtype
        self.encoder: M = self.make_encoder(*args, **kwargs)

    @abstractmethod
    def make_encoder(self, *args, **kwargs) -> M:
        pass

    def prepare(self, sample: I) -> Batched:
        return Batched(torch.tensor(sample, dtype=self.dtype, device=self.device))

    def sample(self, batch_shape=(), mini_batch_size=-1) -> Batched:
        sample: list[torch.Tensor] = []
        batch_size = math.prod(batch_shape)
        if mini_batch_size <= 0:
            mini_batch_size = batch_size

        for i in range(0, batch_size, mini_batch_size):
            mini_batch = np.array(
                [
                    self.space.sample()
                    for _ in range(min(batch_size - i, mini_batch_size))
                ]
            )

            sample.append(self.prepare(mini_batch).data)

        sample = torch.cat(sample)
        return Batched(
            sample.squeeze(dim=0)
            if len(batch_shape) == 0 else
            sample.unflatten(dim=0, sizes=batch_shape)
        )

    def forward(self, sample: Batched) -> torch.FloatTensor:
        return self.encoder(sample.data).unsqueeze(-2)

    def item(self, sample: Batched) -> np.ndarray:
        return sample.data.cpu().numpy()
