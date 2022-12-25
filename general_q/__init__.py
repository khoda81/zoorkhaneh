from importlib.metadata import version

__version__ = version("general_q")

from . import agents, encoders, utils

__all__ = ["agents", "encoders", "utils"]
