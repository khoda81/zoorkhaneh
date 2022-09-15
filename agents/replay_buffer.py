import torch

from encoder.encoder import Encoder


class ReplayMemory:
    def __init__(
            self,
            observation_encoder: Encoder,
            action_encoder: Encoder,
            capacity: int,
    ):
        self.observations = observation_encoder.sample(capacity)
        self.actions = action_encoder.sample(capacity)
        self.rewards = torch.zeros(
            (capacity,), dtype=torch.float32,
            device=observation_encoder.device
        )

        self.terminations = torch.ones(
            (capacity,), dtype=torch.bool,
            device=observation_encoder.device
        )

        self.truncations = torch.ones(
            (capacity,), dtype=torch.bool,
            device=observation_encoder.device
        )

        self.capacity = capacity
        self.last = capacity - 1
        self.size = 0

    def append(self, new_observation, action, reward=0., termination=False, truncation=False) -> None:
        # TODO profile append
        self.last = (self.last + 1) % self.capacity

        self.observations[self.last] = self.observations.encoder.prepare(new_observation)
        self.actions[self.last] = self.actions.encoder.prepare(action)
        self.rewards[self.last] = reward
        self.terminations[self.last] = termination
        self.truncations[self.last] = truncation

        self.size = min(self.size + 1, self.capacity)

    def valid_indices(self):
        valid = ~self.terminations & ~self.truncations
        valid[self.last] = False

        return torch.where(valid)[0]

    def sample(self, batch_size: int):
        indices = self.valid_indices()
        indices = indices[torch.randperm(len(indices))]
        indices = indices[:batch_size]

        return self[indices]

    def close(self):
        self.truncations[self.last] = True

    def __len__(self):
        return len(self.valid_indices())

    def __getitem__(self, item):
        # convert slices to indices
        next_item = (torch.arange(self.size)[item] + 1) % self.capacity

        obs = self.observations[item]
        action = self.actions[next_item]
        reward = self.rewards[next_item]
        terminate = self.terminations[next_item]
        new_obs = self.observations[next_item]

        return obs, action, reward, terminate, new_obs
