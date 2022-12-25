from typing import Optional

import random
from abc import ABC
from pathlib import Path

from gymnasium import Space
from gymnasium.core import ActType, ObsType

NAMES_PATH = Path(__file__).parent / "names.txt"

with open(NAMES_PATH) as f:
    NAMES = f.read().splitlines()


class Agent(ABC):
    def __init__(
            self,
            action_space: Space[ActType],
            observation_space: Space[ObsType],
            name: Optional[str] = None,
    ) -> None:
        """
        Initialize the class.

        Args:
            action_space: The action space of the environment.
            observation_space: The observation space of the environment.
            name: The name of the agent.
        """
        if name is None:
            name = random.choice(NAMES)

        self.action_space = action_space
        self.observation_space = observation_space
        self.name = name

    def act(self, obs: ObsType) -> tuple[ActType, float]:
        """
        Act based on the observation.

        Args:
            obs (np.ndarray): Observation from the environment.

        Returns:
            The action to take and the value of the action.
        """
        return self.action_space.sample(), 0.0

    def reset(self) -> None:
        """
        Reset the state of the agent. And get ready for next episode.
        """

    def remember(
            self,
            new_observation: ObsType,
            action: Optional[ActType] = None,
            reward: float = 0.0,
            termination: bool = False,
            truncation: bool = False,
    ) -> None:
        """
        Remember the action taken.

        Args:
            new_observation: The observation after the action.
            action: The action taken.
            reward: The reward received.
            termination: Whether the episode is done.
            truncation: Whether the episode is truncated.
        """

    def learn(self) -> float:
        """
        Perform one training step.

        Returns:
            The loss value.
        """
        return 0.0

    def close(self) -> None:
        """Close the agent."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()
        return False

    def __call__(self, obs: ObsType) -> ActType:
        """Call agent.act()"""
        return self.act(obs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name!r})'
