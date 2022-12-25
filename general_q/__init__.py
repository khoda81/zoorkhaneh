# type: ignore[attr-defined]
"""An easy to use library for general purpose reinforcement learning and experimentation"""

from . import agents, encoders, utils


def get_version() -> str:
    from importlib import metadata as importlib_metadata

    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__: str = get_version()
