import warnings

import torch

from general_q.encoders import Encoder


class InvalidMemoryState(Exception):
    pass


class ReplayMemory:
    def __init__(
            self,
            observation_encoder: Encoder,
            action_encoder: Encoder,
            capacity: int,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        # TODO remove observation and action encoders from replay memory
        # TODO remove device
        # fmt: off
        self.observation_encoder = observation_encoder
        self.action_encoder      = action_encoder
        self.observations        = observation_encoder.sample([capacity])
        self.actions             = action_encoder.sample([capacity])
        self.rewards             = torch.zeros([capacity], dtype=torch.float32, device=device)
        self.terminations        = torch.ones([capacity], dtype=torch.bool, device=device)
        self.truncations         = torch.ones([capacity], dtype=torch.bool, device=device)
        # fmt: on

        self.capacity = capacity
        self.last = capacity - 1
        self.size = 0

    def append(
            self,
            new_observation,
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
    ) -> None:
        # TODO profile append
        observation = self.observation_encoder.prepare(new_observation)

        if action is None and None not in self.action_encoder.space:  # TODO What if None is a valid action?
            action = self.action_encoder.sample(batch_shape=())

            term = self.terminations[self.last]
            trunc = self.truncations[self.last]

            if not (term or trunc):
                raise InvalidMemoryState(
                    f"No action was provided meaning this is the initial observation, but "
                    f"the last memory state is {~term * 'non-'}terminal and {~trunc * 'non-'}truncated. "
                )

            if terminated or truncated:
                warnings.warn(
                    f"No action was provided meaning this is the initial observation, but"
                    f"the new state is {~terminated * 'non-'}terminal and {~truncated * 'non-'}truncated. "
                    f"Agent can't be terminated before it has been created!"
                )
        else:
            action = self.action_encoder.prepare(action)

        self.last = (self.last + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # fmt: off
        self.observations[self.last] = observation
        self.actions     [self.last] = action
        self.rewards     [self.last] = reward
        self.terminations[self.last] = terminated
        self.truncations [self.last] = truncated
        # fmt: on

    def valid_indices(self):
        invalid = self.terminations | self.truncations
        invalid[self.last] = True

        indices, = torch.where(~invalid)
        return indices

    def sample(self, batch_size: int):
        indices = self.valid_indices()
        indices = indices[torch.randperm(len(indices))]  # randomize indices
        indices = indices[:batch_size]

        return self[indices]

    def close(self):
        self.truncations[self.last] = True

    def __add__(self, other: "ReplayMemory") -> "ReplayMemory":
        """
        Concatenate two replay memories and return a new with size: s1+s2 and cap=c1+c2
        """
        if self.observation_encoder != other.observation_encoder:
            raise ValueError(
                "Cannot concatenate replay memories with different observation encoders")
        if self.action_encoder != other.action_encoder:
            raise ValueError("Cannot concatenate replay memories with different action encoders")

        capacity = self.capacity + other.capacity
        concatenated = ReplayMemory(
            self.observation_encoder,
            self.action_encoder,
            capacity,
        )

        raise NotImplementedError

    def __bool__(self):
        return len(self) > 0

    def __len__(self):
        return len(self.valid_indices())

    def __getitem__(self, item):
        # TODO implement circular buffer indexing

        # convert slices to indices
        item = torch.arange(self.size)[item]
        next_item = (item + 1) % self.capacity

        # fmt: on
        obs = self.observations[item]
        action = self.actions[next_item]
        reward = self.rewards[next_item]
        terminations = self.terminations[next_item]
        truncations = self.truncations[next_item]
        new_obs = self.observations[next_item]
        # fmt: off

        return obs, action, reward, terminations, truncations, new_obs
