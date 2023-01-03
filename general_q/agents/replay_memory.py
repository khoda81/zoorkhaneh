import torch
from gymnasium import spaces

from general_q.encoders import Encoder
from general_q.encoders.composite_encoders import DictEncoder
from general_q.encoders.tensor_encoder import TensorEncoder


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
        # TODO stop allocating all memory at once, implement dynamic list
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

    def append_initial(self, observation) -> None:
        term = self.terminations[self.last]
        trunc = self.truncations[self.last]

        if not (term or trunc):
            raise InvalidMemoryState(
                f"No action was provided meaning this is the initial observation, but "
                f"the last memory state is {~term * 'non-'}terminal and {~trunc * 'non-'}truncated. "
            )

        observation = self.observation_encoder.prepare(observation)

        self.last = (self.last + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # fmt: off
        self.observations[self.last] = observation
        self.rewards     [self.last] = 0.
        self.terminations[self.last] = False
        self.truncations [self.last] = False
        # fmt: on

    def append_transition(
            self,
            new_observation,
            action,
            reward,
            terminated,
            truncated,
    ) -> None:
        new_observation = self.observation_encoder.prepare(new_observation)
        action = self.action_encoder.prepare(action)

        self.last = (self.last + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # fmt: off
        self.observations[self.last] = new_observation
        self.actions     [self.last] = action
        self.rewards     [self.last] = reward
        self.terminations[self.last] = terminated
        self.truncations [self.last] = truncated
        # fmt: on

    def transition_indices(self):
        invalid = self.terminations | self.truncations
        invalid[self.last] = True

        indices, = torch.where(~invalid)
        indices -= self.last - self.size + 1
        return indices % self.capacity

    def sample_transitions(self, batch_size: int):
        indices = self.transition_indices()
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

        raise NotImplementedError  # TODO

    def __bool__(self):
        return self.size > 0

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        item = torch.arange(self.size)[item]  # convert slices to indices
        item += self.last - self.size + 1
        item %= self.capacity
        next_item = (item + 1) % self.capacity

        # fmt: off
        obs          = self.observations[item]
        action       = self.actions[next_item]
        reward       = self.rewards[next_item]
        terminations = self.terminations[next_item]
        truncations  = self.truncations[next_item]
        new_obs      = self.observations[next_item]
        # fmt: on

        return obs, action, reward, terminations, truncations, new_obs

    def __setitem__(self, key, value):
        raise NotImplementedError  # TODO
