import torch
from gymnasium import spaces
from torch import nn

from general_q.encoders.storage import TensorStorage
from general_q.encoders.tensor_encoder import TensorEncoder


class DiscreteEncoder(TensorEncoder[int]):
    space: spaces.Discrete

    def make_encoder(self, embed_dim):
        encoder = nn.Embedding(self.space.n, embed_dim)

        # fmt: off
        start   = encoder.weight[None, 0]                # [1, embed_dim]
        end     = encoder.weight[None, -1]               # [1, embed_dim]
        weights = torch.linspace(0, 1, self.space.n)     # [num_embeddings]
        weights = weights[:, None]                       # [num_embeddings, 1]
        weights = start * (1 - weights) + end * weights  # [num_embeddings, embed_dim]
        # fmt: on

        encoder.weight.data = weights
        return encoder

    def all(self):
        """Make a batch of all possible values of this encoder"""
        items = self.prepare(range(self.space.n))
        return items, self(items)

    def unprepare(self, sample: TensorStorage) -> int:
        return sample.data.item()
