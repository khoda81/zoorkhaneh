import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from general_q.encoders.tensor_encoder import TensorEncoder


class ImageEncoder(TensorEncoder):
    def make_encoder(self, space, embed_dim):
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=-3),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample = self.sample(1)
            sample = sample.to(torch.float32) / 255.0 - 0.5
            _, n = self(sample).shape

        self.encoder.append(nn.Linear(n, embed_dim))

    @staticmethod
    def supports(space: spaces.Space) -> bool:
        if not (
                type(space) == spaces.Box
                and space.dtype == np.uint8
                and len(space.shape) == 3
                and space.low.min() == 0
                and space.high.max() == 255
        ):
            return False

        h, w, c = space.shape
        return c == 3 and h >= 30 and w >= 30

    def prepare(self, sample):
        # rearrange last three dimensions to be (c, h, w)
        sample = super().prepare(sample)
        sample._data = sample._data.transpose(-1, -3)
        return sample

    def forward(self, x):
        return self.encoder(x.data.to(torch.float32) / 255.0 - 0.5)

    def item(self, x):
        return x.data.transpose(-1, -3).cpu().numpy(dtype=np.uint8)
