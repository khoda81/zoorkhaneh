import numpy as np

from abc import ABC, abstractmethod


class AgentBase(ABC):
    def __init__(self, name: str) -> None:
        """
        Initialize the class.

        Args:
            name: The name of the agent.
        """
        self.name = name

    @abstractmethod
    def act(self, obs: np.ndarray, remember: bool = False) -> int:
        """
        Act based on the observation.

        Args:
            obs (np.ndarray): Observation from the environment.
            remember (bool): Whether to remember the action.

        Returns:
            int: Action to take.
        """
        pass

    def reward(self, reward: float) -> None:
        """
        Reward the agent for its last action.

        Args:
            reward: The reward value.
        """
        pass

    def reset(self) -> None:
        """
        Reset the state of the agent.
        """
        pass

    def remember(self, state, action, reward, done) -> None:
        """
        Remember the action taken.

        Args:
            state: The state before the action.
            action: The action taken.
            reward: The reward received.
            next_state: The state after the action.
            done: Whether the episode is done.
        """
        pass

    def __call__(self, obs, remember=True):
        """
        Call agent.act()

        Args:
            obs (np.ndarray): Observation from the environment.
            remember (bool): Whether to remember the action.

        Returns:
            int: Action to take.
        """
        return self.act(obs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


class RandomAgent(AgentBase):
    def act(self, state: tuple[np.ndarray, bool]) -> int:
        board, my_turn = state
        return np.random.choice(np.where(board == 0)[0])
