import math

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from general_q.encoders.tensor_encoder import TensorEncoder


class BoxEncoder(TensorEncoder):
    def make_encoder(self, embed_dim=256):
        return nn.Linear(math.prod(self.space.shape), embed_dim)

    @classmethod
    def supports(cls, space: spaces.Space) -> bool:
        return type(space) == spaces.Box and space.dtype in [
            np.float16,
            np.float32,
            np.float64,
        ]

    def prepare(self, sample):
        def _prepare(sample: torch.Tensor) -> torch.Tensor:
            batch_shape = sample.shape[:len(sample.shape) - len(self.space.shape)]
            return sample.reshape(batch_shape + (-1,))

        return super().prepare(sample).apply(_prepare)

    def item(self, sample: torch.Tensor):
        *batch_shape, _ = sample.shape
        new_shape = (*batch_shape, *self.space.shape)
        return sample.reshape(new_shape).cpu().numpy()
