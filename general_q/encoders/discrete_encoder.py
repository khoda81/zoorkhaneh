from typing import Generic

from abc import ABC, abstractmethod

import torch
from gymnasium import spaces
from torch import nn

from general_q.encoders.base import B, Encoder, I
from general_q.encoders.tensor_encoder import TensorEncoder


class Discrete(Encoder[I, B], ABC, Generic[I, B]):
    @abstractmethod
    def all(self) -> tuple[B, torch.FloatTensor]:
        """Make a batch of all possible values of this encoder"""


class DiscreteEncoder(TensorEncoder[int, nn.Embedding], Discrete[int, torch.IntTensor]):
    @staticmethod
    def supports(space: spaces.Space) -> bool:
        return type(space) == spaces.Discrete

    def make_encoder(self, embed_dim):
        encoder = nn.Embedding(self.space.n, embed_dim)

        start = encoder.weight[None, 0]
        end = encoder.weight[None, -1]
        weights = torch.linspace(0, 1, encoder.num_embeddings).unsqueeze(1)
        encoder.weight.data = start * (1 - weights) + end * weights
        return encoder

    def all(self):
        items = self.prepare(range(self.space.n))
        return items, self(items)
