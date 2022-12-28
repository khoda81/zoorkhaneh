from typing import Callable

import functools
import operator
from collections import OrderedDict

import torch
from gymnasium import spaces
from torch import nn

from general_q.encoders.base import Batched, Encoder


def dzip(*mappings):
    keys = functools.reduce(
        operator.and_,
        (mapping.keys() for mapping in mappings),
    )

    for k in keys:
        yield k, tuple(mapping[k] for mapping in mappings)


class BatchedMap(OrderedDict[str, Batched], Batched):
    def batched_getitem(self, item):
        return self.map(lambda v: v.batched_getitem(item))

    def batched_setitem(self, item, value: Batched):
        self.dzip(value).map(lambda v: v[0].batched_setitem(item, v[1]))

    def apply(self, func: Callable[[Batched], Batched]) -> Batched:
        return self.__class__({k: v.apply(func) for k, v in self.items()})

    def map(self, func: Callable[[Batched], Batched]) -> Batched:
        return self.__class__({k: func(v) for k, v in self.items()})

    def dzip(self, *mappings):
        return self.__class__(dzip(self, *mappings))

    def __repr__(self):
        return super(OrderedDict, self).__repr__()


class DictEncoder(Encoder, nn.ModuleDict):
    def __init__(
            self,
            space: spaces.Dict,
            subencoder,
            embed_dim=256,
            *args, **kwargs,
    ):
        super().__init__(space, *args, **kwargs)
        for key, subspace in space.items():
            encoder = subencoder(space=subspace, embed_dim=embed_dim, *args, **kwargs)
            assert isinstance(encoder, Encoder), f"{encoder} should inherit from {Encoder}"
            self[key] = encoder

    @property
    def batched(self):
        return BatchedMap(self)

    @staticmethod
    def supports(space: spaces.Space) -> bool:
        return isinstance(space, spaces.Dict)

    def prepare(self, sample: dict):
        return self.batched.dzip(sample).map(lambda e, s: e.prepare(s))

    def sample(self, *args, **kwargs):
        return BatchedMap(self).map(lambda e: e.sample(*args, **kwargs))

    def forward(self, sample):
        encoded = [e(s) for _, (e, s) in dzip(self, sample)]
        return torch.cat(encoded, dim=-2)

    def item(self, sample):
        return OrderedDict((k, e.item(s)) for k, (e, s) in dzip(self, sample))


class BatchedTuple(tuple, Batched):
    def batched_getitem(self, item):
        return self.map(lambda v: v.batched_getitem(item))

    def batched_setitem(self, item, value: Batched):
        for v1, v2 in zip(self, value):
            v1.batched_setitem(item, v2)

    def apply(self, func: Callable[[Batched], Batched]) -> Batched:
        return self.__class__(v.apply(func) for v in self)

    def map(self, func: Callable[[Batched], Batched]) -> Batched:
        return self.__class__(func(v) for v in self)

    def zip(self, *iters):
        return self.__class__(zip(self, *iters))


class TupleEncoder(Encoder, nn.ModuleList):
    def __init__(
            self,
            space: spaces.Tuple,
            subencoder,
            embed_dim=256,
            *args, **kwargs,
    ):
        super().__init__(
            space,
            modules=(
                subencoder(space=space, embed_dim=embed_dim, *args, **kwargs)
                for space in space.spaces
            ),
            *args, **kwargs,
        )

    @staticmethod
    def supports(space: spaces.Space) -> bool:
        return isinstance(space, spaces.Tuple)

    def prepare(self, sample: BatchedTuple):
        return tuple(e.prepare(s) for e, s in zip(self, sample))

    def sample(self, *args, **kwargs):
        return BatchedTuple(e.sample(*args, **kwargs) for e in self)

    def forward(self, sample):
        encoded = [e(s) for e, s in zip(self, sample)]
        return torch.cat(encoded, dim=-2)

    def item(self, sample):
        return tuple(e.item(s) for e, s in zip(self, sample))
