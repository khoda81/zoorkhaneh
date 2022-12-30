from general_q.encoders.base import Encoder, UnsupportedSpaceError
from general_q.encoders.composite_encoders import DictEncoder, TupleEncoder
from general_q.encoders.discrete_encoder import DiscreteEncoder
from general_q.encoders.image_encoder import ImageEncoder
from general_q.encoders.storage import (
    MapStorage,
    Storage,
    TensorStorage,
    TupleStorage,
)
from general_q.encoders.tensor_encoder import FloatTensorEncoder, TensorEncoder

__all__ = [
    'auto_encoder',
    'DictEncoder',
    'DiscreteEncoder',
    'Encoder',
    'FloatTensorEncoder',
    'ImageEncoder',
    'MapStorage',
    'Storage',
    'TensorEncoder',
    'TensorStorage',
    'TupleEncoder',
    'TupleStorage',
    'UnsupportedSpaceError',
]


def auto_encoder(space, embed_dim: int, *args, **kwargs):
    encoders = [
        lambda: ImageEncoder(space, embed_dim=embed_dim, *args, **kwargs),
        lambda: FloatTensorEncoder(space, embed_dim=embed_dim, *args, **kwargs),
        lambda: DiscreteEncoder(space, embed_dim=embed_dim, *args, **kwargs),
        lambda: DictEncoder(space, subencoder=auto_encoder, embed_dim=embed_dim, *args, **kwargs),
        lambda: TupleEncoder(space, subencoder=auto_encoder, embed_dim=embed_dim, *args, **kwargs),
    ]

    for make_function in encoders:
        try:
            return make_function()
        except UnsupportedSpaceError:
            pass

    raise AssertionError(f"No encoder for {space}, consider using a different subencoder")


def main():
    from gymnasium.spaces import Box, Dict, Discrete, Tuple

    # fmt: off
    space = Dict(
        {
            "ext_controller": Tuple([Discrete(5), Discrete(3), Discrete(2)]),
            "inner_state": Dict(
                {
                    "charge": Discrete(100),
                    "system_checks": Tuple([Discrete(2)] * 10),
                    "job_status": Dict(
                        {
                            "task": Discrete(5),
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
