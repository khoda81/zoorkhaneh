from typing import Generic

from abc import ABC, abstractmethod

import torch
from gym import spaces
from torch import nn

from general_q.encoders.base import Encoder, I, T
from general_q.encoders.tensor_encoder import TensorBatch, TensorEncoder


class Discrete(Encoder[T, I], ABC, Generic[T, I]):
    @abstractmethod
    def all(self) -> tuple[T, torch.FloatTensor]:
        """Make a batch of all possible values of this encoder"""


class DiscreteEncoder(TensorEncoder[nn.Embedding, int], Discrete[TensorBatch, int]):
    @staticmethod
    def supports(space: spaces.Space) -> bool:
        return type(space) == spaces.Discrete

    @torch.no_grad()
    def make_encoder(self, space: spaces.Discrete, embed_dim) -> nn.Embedding:
        encoder = nn.Embedding(space.n, embed_dim)

        start = encoder.weight[None, 0]
        end = encoder.weight[None, -1]
        weights = torch.linspace(0, 1, encoder.num_embeddings).unsqueeze(1)
        encoder.weight.data = start * (1 - weights) + end * weights
        return encoder

    def all(self):
        items = TensorBatch(torch.arange(self.encoder.num_embeddings, device=self.device), self)
        return items, self(items)

    def item(self, sample):
        return sample.data.item()


class MultiDiscreteEncoder(TensorEncoder[nn.ModuleList, I]):
    @staticmethod
    def supports(space) -> bool:
        return type(space) == spaces.MultiDiscrete

    def make_encoder(self, space, embed_dim):
        return nn.ModuleList([
            DiscreteEncoder(spaces.Discrete(n), embed_dim)
            for n in space.nvec
        ])

    def forward(self, x):
        x = x.data
        return torch.sum(
            torch.stack(
                [encoder(x[..., i]) for i, encoder in enumerate(self.encoder)],
                dim=-1,
            ),
            dim=-1,
        )


class MultiBinaryEncoder(TensorEncoder):
    @staticmethod
    def supports(space: spaces.Space) -> bool:
        return type(space) == spaces.MultiBinary

    def make_encoder(self, space, embed_dim):
        class Float(nn.Module):
            def forward(self, x):
                return x.float()

        return nn.Sequential(
            Float(),
            nn.Flatten(start_dim=-len(space.shape)),
            nn.Linear(space.n, embed_dim),
        )
