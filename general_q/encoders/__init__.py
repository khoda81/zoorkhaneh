from typing import Callable

from gymnasium import spaces

from general_q.encoders.base import B, Batched, BatchProxy, Encoder, I
from general_q.encoders.box_encoder import BoxEncoder
from general_q.encoders.composite_encoders import (
    BatchedMap,
    BatchedTuple,
    DictEncoder,
    TupleEncoder,
)
from general_q.encoders.discrete_encoder import DiscreteEncoder
from general_q.encoders.image_encoder import ImageEncoder
from general_q.encoders.tensor_encoder import TensorEncoder

__all__ = [
    'auto_encoder',
    'Batched', 
    'BatchProxy',
    'BatchedMap',
    'BatchedTuple',
    'BoxEncoder',
    'ImageEncoder',
    'DictEncoder',
    'DiscreteEncoder',
    'Encoder',
    'TensorEncoder',
    'TupleEncoder',
]


def auto_encoder(space: spaces.Space[I], embed_dim: int, *args, **kwargs) -> Encoder[I, B]:
    encoders: Callable[[], Encoder[I, B]] = [
        lambda: ImageEncoder(space, embed_dim=embed_dim),
        lambda: BoxEncoder(space, embed_dim=embed_dim),
        lambda: DiscreteEncoder(space, embed_dim=embed_dim),
        lambda: DictEncoder(space, subencoder=auto_encoder, embed_dim=embed_dim),
        lambda: TupleEncoder(space, subencoder=auto_encoder, embed_dim=embed_dim),
    ]

    for make_function in encoders:
        try:
            return make_function()
        except AssertionError:
            pass

    raise AssertionError(f"No encoder for {space}, consider using a different subencoder")


def main():
    from gymnasium.spaces import Box, Dict, Discrete, Tuple

    # fmt: off
    space = Dict(
        {
            "ext_controller": Tuple([Discrete(5), Discrete(3), Discrete(2)]),
            "inner_state":    Dict(
                {
                    "charge":        Discrete(100),
                    "system_checks": Tuple([Discrete(2)] * 10),
                    "job_status":    Dict(
                        {
                            "task":     Discrete(5),
                            "progress": Box(low=0, high=100, shape=()),
                        }
                    ),
                }
            ),
        }
    )
    # fmt: on

    encoder = auto_encoder(space, 64)
    sample = encoder.prepare(space.sample())
    encoded = encoder(sample)
    print(encoded.shape)

    encoded = encoder(encoder.sample(10))
    print(encoded.shape)


if __name__ == "__main__":
    main()
