import torch

from general_q.encoders import Encoder


class ReplayMemory:
    def __init__(
            self,
            observation_encoder: Encoder,
            action_encoder: Encoder,
            capacity: int,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.observations = observation_encoder.sample(capacity)
        self.actions = action_encoder.sample(capacity)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.terminations = torch.ones((capacity,), dtype=torch.bool, device=device)
        self.truncations = torch.ones((capacity,), dtype=torch.bool, device=device)

        self.capacity = capacity
        self.last = capacity - 1
        self.size = 0

    def push(
            self,
            new_observation,
            action,
            reward=0.0,
            termination=False,
            truncation=False,
    ) -> None:
        # TODO profile push
        self.last = (self.last + 1) % self.capacity

        self.observations[self.last] = \
            self.observations.encoder.prepare(new_observation)
        self.actions[self.last] = self.actions.encoder.prepare(action)
        self.rewards[self.last] = reward
        self.terminations[self.last] = termination
        self.truncations[self.last] = truncation

        self.size = min(self.size + 1, self.capacity)

    def valid_indices(self):
        is_valid = ~self.terminations & ~self.truncations
        is_valid[self.last] = False

        indices, = torch.where(is_valid)
        return indices

    def sample(self, batch_size: int):
        indices = self.valid_indices()
        indices = indices[torch.randperm(len(indices))]  # randomize indices
        indices = indices[:batch_size]

        return self[indices]

    def close(self):
        self.truncations[self.last] = True

    def __len__(self):
        return len(self.valid_indices())

    def __getitem__(self, item):
        # convert slices to indices
        item = torch.arange(self.size)[item]
        next_item = (item + 1) % self.capacity

        obs = self.observations[item]
        action = self.actions[next_item]
        reward = self.rewards[next_item]
        terminations = self.terminations[next_item]
        truncations = self.truncations[next_item]
        new_obs = self.observations[next_item]

        return obs, action, reward, terminations, truncations, new_obs
