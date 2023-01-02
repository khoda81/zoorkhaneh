from abc import ABC, abstractmethod

import torch
from gymnasium import Space, spaces
from torch import distributions as D
from torch import nn


class Decoder(ABC, nn.Module):
    def __init__(self, space: Space):
        super().__init__()
        self.space = space

    @abstractmethod
    def forward(self, emb: torch.Tensor) -> D.Distribution:
        """Decode the input into a distribution of self.space."""


class DiscreteDecoder(Decoder):
    def __init__(self, space: spaces.Discrete, embed_dim: int):
        super().__init__(space)
        self.decoder = nn.Linear(embed_dim, space.n)

    def forward(self, emb: torch.Tensor) -> D.Categorical:
        return D.Categorical(logits=self.decoder(emb))
