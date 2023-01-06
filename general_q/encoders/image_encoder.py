import numpy as np
import torch
from gymnasium import Space, spaces
from numpy.typing import NDArray
from torch import nn

from general_q.encoders.storage import TensorStorage
from general_q.encoders.tensor_encoder import TensorEncoder


class ImageEncoder(TensorEncoder):
    def make_encoder(self, embed_dim):
        # TODO implement a more robust to size encoder, maybe a pretrained ViT?
        self.encoder = nn.Sequential(
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
            _, n = self.forward(self.sample()).shape
            self.encoder.append(nn.Linear(n, embed_dim))

        return self.encoder

    @staticmethod
    def supports(space: Space) -> bool:
        # fmt: off
        if not (
                type(space)      == spaces.Box
            and space.dtype      == np.uint8
            and len(space.shape) == 3
            and space.low.min()  == 0
            and space.high.max() == 255
        ):
            return False
        # fmt: on

        h, w, c = space.shape
        return c == 3 and h >= 30 and w >= 30

    def prepare(self, sample):
        sample = super().prepare(sample)
        # [*b, h*w*c] -> [*b, c, w, h] -> [*b, c, h, w]
        sample.data = sample.data \
            .reshape(sample.shape[:-1] + self.space.shape) \
            .transpose(-1, -3).transpose(-1, -2)

        return sample

    def unprepare(self, sample: TensorStorage) -> NDArray:
        # [*b, c, h, w] -> [*b, c, w, h] -> [*b, h, w, c]
        return super().unprepare(sample).transpose(-1, -2).transpose(-1, -3)

    def sample(self, batch_shape=()) -> TensorStorage:
        shape = (*batch_shape, *self.space.shape)
        data = torch.randint(0, 255, shape, dtype=self.dtype, device=self.device)
        return self.prepare(data)

    def forward(self, sample: TensorStorage) -> torch.Tensor:
        # TODO add a proper normalizer

        # we normalize only when forwarding to save memory
        # floats take more than 4x the space of uint8s
        sample = sample.transformed(lambda x: x / 255 - .5)
        return super().forward(sample)
