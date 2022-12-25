import math

import numpy as np
from gym import spaces
from torch import nn

from general_q.encoders.tensor_encoder import TensorEncoder


class BoxEncoder(TensorEncoder):
    def make_encoder(self, space: spaces.Box, embed_dim=256):
        class Unsqueeze(nn.Module):
            def forward(self, x):
                return x.unsqueeze(-1)

        preprocess_layer = (
            Unsqueeze()
            if space.shape == ()
            else nn.Flatten(start_dim=-len(space.shape))
        )
        return nn.Sequential(
            preprocess_layer,
            nn.Linear(math.prod(space.shape), embed_dim),
        )

    @classmethod
    def supports(cls, space: spaces.Space) -> bool:
        return type(space) == spaces.Box and space.dtype in [
            np.float16,
            np.float32,
            np.float64,
        ]
