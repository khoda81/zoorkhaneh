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

    def act(self, obs):
        """
        Act based on the observation.

        Args:
            obs (np.ndarray): Observation from the environment.
            remember (bool): Whether to remember the action.

        Returns:
            action Action to take.
        """
        return self.action_space.sample()

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

    def remember(self, new_obs, action=None, reward=0, done=False) -> None:
        """
        Remember the action taken.

        Args:
            new_obs: The observation after the action.
            action: The action taken.
            reward: The reward received.
            done: Whether the episode is done.
        """

    def learn(self) -> float:
        """
        Perform one training step.
        
        Returns:
            The loss value.
        """
        return 0.

    def __call__(self, *args, **kwargs):
        """Call agent.act()"""
        return self.act(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"
