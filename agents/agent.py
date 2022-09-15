from typing import Any

from gym import Space


class AgentBase:
    def __init__(self, action_space: Space, observation_space: Space, name: str) -> None:
        """
        Initialize the class.

        Args:
            action_space: The action space of the environment.
            observation_space: The observation space of the environment.
            name: The name of the agent.
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.name = name

    def act(self, obs) -> tuple[Any, float]:
        """
        Act based on the observation.

        Args:
            obs (np.ndarray): Observation from the environment.

        Returns:
            The action to take and the value of the action.
        """
        return self.action_space.sample(), 0.

    def reward(self, reward: float) -> None:
        """
        Reward the agent for its last action.

        Args:
            reward: The reward value.
        """

    def reset(self) -> None:
        """
        Reset the state of the agent.
        """

    def remember(self, new_observation, action=None, reward=0, termination=False, truncation=False) -> None:
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
        return 0.

    def close(self) -> None:
        """Close the agent."""

    def __call__(self, *args, **kwargs):
        """Call agent.act()"""
        return self.act(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"
