from typing import Generic, Optional

import random
from pathlib import Path

from gymnasium import Space
from gymnasium.core import ActType, ObsType

NAMES_PATH = Path(__file__).parent / "names.txt"

with open(NAMES_PATH) as f:
    NAMES = f.read().splitlines()


class Agent(Generic[ActType, ObsType]):
    # TODO write interface for switching the same agent between different environments
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
        super().__init__()

        if name is None:
            name = random.choice(NAMES)

        self.action_space = action_space
        self.observation_space = observation_space
        self.name = name

    def act(self, obs: ObsType) -> ActType:
        """
        Act based on the observation.

        Args:
            obs: Observation from the environment.

        Returns:
            The action to take.
        """
        return self.action_space.sample()

    def reset(self) -> None:
        """
        Reset the state of the agent and get ready for next episode.
        """

    def remember_initial(self, observation: ObsType) -> None:
        """
        Remember this initial observation for training purposes.

        Args:
            observation: The initial observation.
        """

    def remember_transition(
            self,
            new_observation: ObsType,
            action: ActType,
            reward: float,
            termination: bool,
            truncation: bool,
    ) -> None:
        """
        Remember this action and the consequences. Data stored by this method should only be used for learning.
        Agent state should be stored in object attributes and managed by `agent.act` and `agent.reset`.

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()

    def __call__(self, obs: ObsType) -> ActType:
        """Call agent.act()"""
        return self.act(obs)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name!r})'
