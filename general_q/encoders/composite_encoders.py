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
            embed_dim,
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

    def prepare(self, sample: dict, *args, **kwargs):
        return MapStorage(
            (k, e.prepare(s, *args, **kwargs))
            for k, (e, s) in dzip(self, sample)
        )

    def unprepare(self, sample: MapStorage, *args, **kwargs):
        return {
            k: e.item(s, *args, **kwargs)
            for k, (e, s) in dzip(self, sample.map)
        }

    def sample(self, *args, **kwargs) -> MapStorage:
        return MapStorage(
            (k, e.sample(*args, **kwargs))
            for k, e in self.items()
        )

    def forward(self, sample: MapStorage):
        encoded = [e(s) for _, (e, s) in dzip(self, sample.map)]
        return torch.cat(encoded, dim=-2)


class TupleEncoder(Encoder, nn.ModuleList):
    space: spaces.Tuple

    def __init__(
            self,
            space: spaces.Tuple,
            subencoder,
            embed_dim,
            *args, **kwargs,
    ):
        super().__init__(space, *args, **kwargs)
        for space in space.spaces:
            encoder = subencoder(
                space=space,
                embed_dim=embed_dim,
                *args, **kwargs
            )

            assert isinstance(encoder, Encoder), \
                f"{encoder} should inherit from {Encoder}"

            self.append(encoder)

    @property
    def batched(self):
        return TupleStorage(self)

    def prepare(self, sample):
        return TupleStorage(e.prepare(s) for e, s in zip(self, sample))

    def unprepare(self, sample: TupleStorage):
        return tuple(e.item(s) for e, s in zip(self, sample.items))

    def sample(self, *args, **kwargs):
        return TupleStorage(e.sample(*args, **kwargs) for e in self)

    def forward(self, sample: TupleStorage):
        encoded = [e(s) for e, s in zip(self, sample.items)]
        return torch.cat(encoded, dim=-2)
