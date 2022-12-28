import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from general_q.encoders.base import Batched
from general_q.encoders.tensor_encoder import TensorEncoder


class ImageEncoder(TensorEncoder):
    def make_encoder(self, embed_dim):
        # TODO implement a more robust to size encoder, maybe a Vision Transformer?
        encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=-3),
        )

        with torch.no_grad():
            # sample a single item to get the output shape
            _, n = self(self.sample()).shape
            encoder.append(nn.Linear(n, embed_dim))

        return encoder

    @staticmethod
    def supports(space: spaces.Space) -> bool:
        # fmt: off
        if not (
                type(space) == spaces.Box
                and space.dtype == np.uint8
                and len(space.shape) == 3
                and space.low.min() == 0
                and space.high.max() == 255
        ):
            return False
        # fmt: on

        h, w, c = space.shape
        return c == 3 and h >= 30 and w >= 30

    def prepare(self, sample):
        # byte[h, w, c] -> float[c, h, w]
        return super().prepare(sample).transpose(-1, -3).div(255).sub(0.5)

    def item(self, sample: Batched) -> np.ndarray:
        # float[c, h, w] -> byte[h, w, c]
        return sample.data.add(0.5).mul(255).to(torch.uint8).cpu().numpy()
