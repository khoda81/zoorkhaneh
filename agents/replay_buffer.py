import torch

from agents.encoder import Encoder


class ReplayMemory:
    def __init__(
        self,
        observation_encoder: Encoder,
        action_encoder: Encoder,
        capacity: int,
    ):
        self.observation_encoder = observation_encoder
        self.action_encoder = action_encoder

        self.observations = self.observation_encoder.sample(capacity)
        self.actions = self.action_encoder.sample(capacity)
        self.rewards = torch.zeros(
            (capacity,), dtype=torch.float32,
            device=self.observation_encoder.device
        )

        self.terminations = torch.ones(
            (capacity,), dtype=bool,
            device=self.observation_encoder.device
        )

        self.truncations = torch.ones(
            (capacity,), dtype=bool,
            device=self.observation_encoder.device
        )

        self.last = self.capacity = capacity
        self.size = 0

    def append(self, new_observation, action, reward=0., termination=False, truncation=False) -> None:
        # TODO profile append
        self.last = (self.last + 1) % self.capacity
        
        new_observation = self.observation_encoder.prepare(new_observation)
        self.observation_encoder.setitem(self.observations, self.last, new_observation)

        action = self.action_encoder.prepare(action)
        self.action_encoder.setitem(self.actions, self.last, action)

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
        # convert sclices to indices
        next_item = (torch.arange(self.size)[item] + 1) % self.capacity

        obs = self.observation_encoder.getitem(self.observations, item)
        action = self.action_encoder.getitem(self.actions, next_item)
        reward = self.rewards[next_item]
        terminate = self.terminations[next_item]
        new_obs = self.observation_encoder.getitem(self.observations, next_item)

        return obs, action, reward, terminate, new_obs
