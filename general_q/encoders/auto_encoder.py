from typing import Callable

from gym import spaces

from general_q.encoders.base import Encoder, I, T
from general_q.encoders.box_encoder import BoxEncoder
from general_q.encoders.discrete_encoders import (
    DiscreteEncoder,
    MultiBinaryEncoder,
    MultiDiscreteEncoder,
)
from general_q.encoders.image_encoder import ByteImageEncoder
from general_q.encoders.multi_encoders import DictEncoder, TupleEncoder


def auto_encoder(space: spaces.Space[I], embed_dim: int, *args, **kwargs) -> Encoder[T, I]:
    encoders: Callable[[], Encoder[T, I]] = [
        lambda: ByteImageEncoder(space, embed_dim=embed_dim),
        lambda: BoxEncoder(space, embed_dim=embed_dim),
        lambda: DiscreteEncoder(space, embed_dim=embed_dim),
        lambda: MultiDiscreteEncoder(space, embed_dim=embed_dim),
        lambda: MultiBinaryEncoder(space, embed_dim=embed_dim),
        lambda: DictEncoder(space, subencoder=auto_encoder, embed_dim=embed_dim),
        lambda: TupleEncoder(space, subencoder=auto_encoder, embed_dim=embed_dim),
    ]

    for make_funciton in encoders:
        try:
            return make_funciton()
        except AssertionError:
            pass

    raise ValueError(f"No encoder for {space}, consider using a different subencoder")


def main():
    from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

    space = Dict(
        {
            "ext_controller": MultiDiscrete([5, 2, 2]),
            "inner_state":    Dict(
                {
                    "charge":        Discrete(100),
                    "system_checks": MultiBinary(10),
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

    encoder = auto_encoder(space, 64)
    sample = space.sample()
    new_sample = encoder.prepare(sample).item()

    encoded = encoder(encoder.sample(10))
    print(encoded.shape)


if __name__ == "__main__":
    main()
