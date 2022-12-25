from collections import OrderedDict
from shutil import ExecError

import torch
from gymnasium import spaces
from torch import nn

from general_q.encoders.base import Batch, Encoder


class MultiBatch(Batch[OrderedDict, OrderedDict]):
    @classmethod
    def from_dict(cls, data: OrderedDict, encoder, func=lambda x: x):
        return cls(
            OrderedDict(
                (key, func(value))
                for key, value in data.items()
            ),
            encoder
        )

    def __getitem__(self, item):
        return OrderedDict(
            (k, v[item])
            for k, v in self.data.items()
        )

    def __setitem__(self, key, value):
        for k, v in value.items():
            self._data[k][key] = v


class DictEncoder(Encoder, nn.ModuleDict):
    def __init__(self, space: spaces.Dict, subencoder=lambda: None, embed_dim=256):
        super().__init__(space)
        for key, subspace in space.spaces.items():
            encoder = subencoder(subspace, name=key, embed_dim=embed_dim)

            assert isinstance(encoder, Encoder), f"{encoder} should inherit from {Encoder}"
            self[key] = encoder

    @staticmethod
    def supports(space: spaces.Space) -> bool:
        return isinstance(space, spaces.Dict)

    def prepare(self, sample):
        return MultiBatch(
            OrderedDict(
                (key, encoder.prepare(sample[key]))
                for key, encoder in self.items()
            ),
            self
        )

    def sample(self, batch_size=1):
        return MultiBatch.from_dict(self, self, lambda encoder: encoder.sample(batch_size))

    def forward(self, x):
        return torch.sum(
            torch.stack([encoder(x.data[key]) for key, encoder in self.items()], dim=-1),
            dim=-1,
        )

    def item(self, sample):
        return OrderedDict(
            (key, encoder.item(sample.data[key]))
            for key, encoder in self.items()
        )

    def atomic_encoders(self):
        for key, encoder in self.items():
            for sub_key, sub_encoder in encoder.atomic_encoders():
                yield (key, *sub_key), sub_encoder

class TupleEncoder(DictEncoder):
    def __init__(self, space: spaces.Tuple, embed_dim=256):
        space = spaces.Dict(enumerate(space.spaces))
        super().__init__(space, embed_dim)

    @staticmethod
    def supports(space: spaces.Space) -> bool:
        return isinstance(space, spaces.Tuple)

    def prepare(self, sample):
        return super().prepare(OrderedDict(enumerate(sample)))

    def item(self, sample):
        return tuple(super().item(sample).values())
