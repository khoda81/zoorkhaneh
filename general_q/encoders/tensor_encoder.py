from typing import Generic, Optional, TypeVar

import math

import numpy as np
import torch
from gymnasium import Space, spaces
from numpy.typing import NDArray
from torch import nn

from general_q.encoders.base import Encoder, I
from general_q.encoders.storage import TensorStorage

M = TypeVar("M", bound=torch.nn.Module)


def unflatten(tensor: torch.Tensor, dim, sizes):
    if len(sizes) == 0:
        return tensor.squeeze(dim=dim)
    else:
        return tensor.unflatten(dim=dim, sizes=sizes)


class TensorEncoder(Encoder[I, TensorStorage], Generic[I]):
    def __init__(self, space: Space[I], embed_dim: Optional[int], *args, **kwargs):
        super().__init__(space)
        sample = torch.tensor(space.sample())
        self.dtype = sample.dtype
        # TODO remove assert after done with testing
        assert sample.shape == self.space.shape, \
            f"Sample's shape={tuple(sample.shape)} does not match expected shape: " \
            f"space.shape={', '.join(map(str, self.space.shape))}, " \
            f"space={space}"

        if embed_dim is not None:
            self.encoder = self.make_encoder(embed_dim, *args, **kwargs)

    def make_encoder(self, *args, **kwargs) -> nn.Module:
        raise NotImplementedError("If you don't need a neural network, "
                                  "set embed_dim=None in the constructor")

    def prepare(self, sample: I) -> TensorStorage:
        if isinstance(sample, torch.Tensor):
            sample = sample.to(self.device, self.dtype)
        else:
            sample = torch.tensor(sample, dtype=self.dtype, device=self.device)

        num_space_dims = len(self.space.shape)
        num_sample_shape = len(sample.shape)
        num_batch_dims = num_sample_shape - num_space_dims
        assert sample.shape[num_batch_dims:] == self.space.shape, \
            f"Sample shape={tuple(sample.shape)} does not match expected shape: " \
            f"(*batch_shape, {', '.join(map(str, self.space.shape))})"

        if num_space_dims > 0:
            sample = sample.flatten(start_dim=num_batch_dims)

        return TensorStorage(sample)

    def unprepare(self, sample: TensorStorage) -> NDArray:
        return sample.data.cpu().numpy()

    def sample(self, batch_shape=()) -> TensorStorage:
        batch_size = math.prod(batch_shape)

        sample = np.array([self.space.sample() for _ in range(batch_size)])
        sample = torch.from_numpy(sample).to(self.device)
        sample = unflatten(sample, dim=0, sizes=batch_shape)

        return self.prepare(sample)

    def forward(self, sample: TensorStorage) -> torch.FloatTensor:
        if not hasattr(self, "encoder"):
            raise ValueError(
                "There is no model to encode the sample. "
                "This might be because encoder was created with embed_dim=None."
            )

        return self.encoder(sample.data).unsqueeze(-2)


class DiscreteEncoder(TensorEncoder[int]):
    space: spaces.Discrete

    def make_encoder(self, embed_dim):
        encoder = nn.Embedding(self.space.n, embed_dim)

        # make embeddings linearly spaced between first and last embedding
        # this is useful for encoding order in untrained embeddings
        start   = encoder.weight[None, 0]                # [1, embed_dim]
        end     = encoder.weight[None, -1]               # [1, embed_dim]
        weights = torch.linspace(0, 1, self.space.n)     # [num_embeddings]
        weights = weights[:, None]                       # [num_embeddings, 1]
        weights = start * (1 - weights) + end * weights  # [num_embeddings, embed_dim]

        encoder.weight.data = weights
        return encoder

    def all(self):
        """Make a batch of all possible values of this encoder"""
        items = self.prepare(range(self.space.n))
        return items, self(items)

    def unprepare(self, sample: TensorStorage) -> int:
        return sample.data.item()

    def sample(self, batch_shape):
        return self.prepare(torch.randint(self.space.n, batch_shape))


class FloatTensorEncoder(TensorEncoder):
    # TODO add normalize option
    space: spaces.Box

    @classmethod
    def supports(cls, space: Space) -> bool:
        dtypes = [np.float16, np.float32, np.float64]
        return super().supports(space) and space.dtype in dtypes

    def make_encoder(self, embed_dim):
        return nn.Linear(math.prod(self.space.shape), embed_dim)

    def sample(self, batch_shape=()) -> TensorStorage:
        batch_shape = tuple(batch_shape)
        repeats = batch_shape + (1,) * len(self.space.shape)

        space = spaces.Box(
            low=np.tile(self.space.low,  repeats),
            high=np.tile(self.space.high, repeats),
            shape=batch_shape + self.space.shape,
            dtype=self.space.dtype,
        )

        return self.prepare(space.sample())


class IntTensorEncoder(TensorEncoder):
    space: spaces.Box
    max_discrete_count = 2**16

    def __init__(self, space: spaces.Box, embed_dim=None, *args, **kwargs):
        super().__init__(space, embed_dim, *args, **kwargs)
        low = torch.tensor(space.low).flatten()
        self.register_buffer("low", low)

        high   = torch.tensor(space.high).flatten()
        bounds = high + 1 - low
        coefs  = torch.cumprod(bounds.flipud(), dim=0)
        self.n = coefs[-1].item()
        coefs  = torch.cat([torch.tensor([1]), coefs[:-1]]).flipud()
        self.register_buffer("coefs", coefs)

    @classmethod
    def supports(cls, space: Space) -> bool:
        dtypes = [np.int8, np.int16, np.int32, np.int64]

        if super().supports(space) and space.dtype in dtypes:
            bounds = (space.high - space.low).flatten()
            n = 1
            for bound in bounds:
                n *= int(bound)
                if n > cls.max_discrete_count:
                    return False

            return True

        return False

    def make_encoder(self, *args, **kwargs):
        return DiscreteEncoder(spaces.Discrete(self.n), *args, **kwargs)

    def prepare(self, sample):
        sample = super().prepare(sample)

        sample.data.sub_(self.low)            # [*batch_shape, prod(space.shape)]
        sample.data.unsqueeze_(0)             # [1, *batch_shape, prod(space.shape)]
        sample.data.transpose_(0, -1)         # [prod(space.shape), *batch_shape, 1]

        sample.data.mul_(self.coefs)          # [prod(space.shape), *batch_shape, 1]
        sample.data = sample.data.sum(dim=0)  # [*batch_shape, 1]

        return sample

    def sample(self, batch_shape=()) -> TensorStorage:
        batch_shape = tuple(batch_shape)
        repeats = batch_shape + (1,) * len(self.space.shape)

        space = spaces.Box(
            low=np.tile(self.space.low,  repeats),
            high=np.tile(self.space.high, repeats),
            shape=batch_shape + self.space.shape,
            dtype=self.space.dtype,
        )

        return self.prepare(space.sample())
