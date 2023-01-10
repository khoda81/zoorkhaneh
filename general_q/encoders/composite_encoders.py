from typing import Optional

import torch
from gymnasium import spaces
from torch import nn

from general_q.encoders.base import Encoder
from general_q.encoders.storage import MapStorage, TupleStorage, dzip


class DictEncoder(Encoder, nn.ModuleDict):
    space: spaces.Dict

    def __init__(
            self,
            space: spaces.Dict,
            subencoder,
            embed_dim: Optional[int],
            *args, **kwargs,
    ):
        super().__init__(space, *args, **kwargs)
        for key, subspace in space.items():
            encoder = subencoder(
                space=subspace,
                embed_dim=embed_dim,
                *args, **kwargs
            )

            assert isinstance(encoder, Encoder), \
                f"{encoder} should inherit from {Encoder}"

            self[key] = encoder

    def prepare(self, sample: dict):
        return MapStorage((k, e.prepare(s)) for k, (e, s) in dzip(self, sample, keys=sample))

    def unprepare(self, sample: MapStorage):
        return {k: e.item(s) for k, (e, s) in dzip(self, sample.map, keys=sample.map)}

    def sample(self, batch_shape=()) -> MapStorage:
        return MapStorage((k, e.sample(batch_shape=batch_shape)) for k, e in self.items())

    def forward(self, sample: MapStorage):
        encoded = [e(s) for _, (e, s) in dzip(self, sample.map, keys=self)]
        return torch.cat(encoded, dim=-2)


class TupleEncoder(Encoder, nn.ModuleList):
    space: spaces.Tuple

    def __init__(
            self,
            space: spaces.Tuple,
            subencoder,
            embed_dim: Optional[int],
            *args, **kwargs,
    ):
        super().__init__(space, *args, **kwargs)
        for space in space.spaces:
            encoder = subencoder(
                space=space,
                embed_dim=embed_dim,
                *args, **kwargs,
            )

            assert isinstance(encoder, Encoder), \
                f"{encoder} should inherit from {Encoder}"

            self.append(encoder)

    def prepare(self, sample):
        return TupleStorage(e.prepare(s) for e, s in zip(self, sample))

    def unprepare(self, sample: TupleStorage):
        return tuple(e.item(s) for e, s in zip(self, sample.items))

    def sample(self, batch_shape=()):
        return TupleStorage(e.sample(batch_shape=batch_shape) for e in self)

    def forward(self, sample: TupleStorage):
        encoded = [e(s) for e, s in zip(self, sample.items)]
        return torch.cat(encoded, dim=-2)
