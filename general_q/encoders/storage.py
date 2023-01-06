from typing import Any, Callable

from abc import abstractmethod, abstractproperty
from collections import OrderedDict
from collections.abc import Hashable

import torch


class Storage:
    @abstractmethod
    def apply(self, transform: Callable[[Any], Any]) -> "Storage":
        pass

    @abstractproperty
    def shape(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __setitem__(self, item, value):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class TensorStorage(Storage):
    def __init__(self, data: torch.Tensor):
        self.data = data

    @property
    def shape(self):
        return self.data.shape

    def apply(self, transform):
        return self.__class__(transform(self.data))

    def __getitem__(self, item):
        return self.__class__(self.data[item])

    def __iter__(self):
        return map(self.__class__, self.data)

    def __setitem__(self, item, value: "TensorStorage"):
        self.data[item] = value.data

    def __repr__(self):
        return self.data.__repr__()


def dzip(*mappings, keys=None, collect=tuple):
    if not mappings:
        if keys is not None:
            yield from (
                (key, collect(iter(())))
                for key in keys
            )

        return

    if keys is None:
        keys = mappings[0]

    for key in keys:
        yield key, collect(mapping[key] for mapping in mappings)


class MapStorage(Storage):
    map: OrderedDict[Hashable, Storage]

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.map = OrderedDict(*args, **kwargs)

    def apply(self, transform):
        return self.__class__(
            (k, v.apply(transform))
            for k, v in self.map.items()
        )

    @property
    def shape(self):
        return {k: v.shape for k, v in self.map.items()}

    def __repr__(self):  # TODO write a proper repr
        return self.map.__repr__()

    def __getitem__(self, item):
        return self.__class__(
            (k, v[item])
            for k, v in self.map.items()
        )

    def __setitem__(self, item, value: "MapStorage"):
        for _, (v1, v2) in dzip(self.map, value.map, keys=value.map):
            v1[item] = v2

    def __delitem__(self, item):
        for v in self.map.values():
            del v[item]

    def __iter__(self):
        iters = {k: iter(v) for k, v in self.map.items()}
        while True:
            try:
                yield self.__class__((k, next(iters[k])) for k in iters)
            except StopIteration:
                return


class TupleStorage(Storage):
    items: tuple[Storage]

    def __init__(self, data):
        super().__init__()
        self.items = tuple(data)

    def apply(self, transform):
        return self.__class__(v.apply(transform) for v in self.items)

    @property
    def shape(self):
        return tuple(v.shape for v in self.items)

    def __repr__(self):
        return self.items.__repr__()

    def __getitem__(self, item):
        return self.__class__(v[item] for v in self.items)

    def __setitem__(self, item, value: "TupleStorage"):
        for v1, v2 in zip(self.items, value.items):
            v1[item] = v2

    def __delitem__(self, item):
        for v in self.items:
            del v[item]

    def __iter__(self):
        return map(self.__class__, zip(*self.items))
