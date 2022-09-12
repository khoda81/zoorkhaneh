import torch

from agents.encoder import Encoder


class ReplayDeque:
    def __init__(
        self,
        observation_encoder: Encoder,
        action_encoder: Encoder,
        max_size: int,
    ):
        self.observation_encoder = observation_encoder
        self.action_encoder = action_encoder

        self.observations = self.observation_encoder.sample(0)
        self.actions = self.action_encoder.sample(0)
        self.rewards = torch.empty(
            (0,),
            dtype=torch.float32,
            device=self.observation_encoder.device
        )
        self.dones = torch.empty(
            (0,),
            dtype=bool,
            device=self.observation_encoder.device
        )

        self.max_size = max_size

    def append(self, new_observation, action, reward, done):
        # TODO profile append
        new_observation = self.observation_encoder.prepare(new_observation)
        new_observation = self.observation_encoder.getitem(new_observation, [None])

        action = self.action_encoder.prepare(action)
        action = self.action_encoder.getitem(action, [None])

        reward = torch.tensor([reward], dtype=torch.float32, device=self.observation_encoder.device)
        done = torch.tensor([done], dtype=bool, device=self.observation_encoder.device)

        self.observations = self.observation_encoder.concat(self.observations, new_observation)
        self.actions = self.action_encoder.concat(self.actions, action)
        self.rewards = torch.cat([self.rewards, reward])
        self.dones = torch.cat([self.dones, done])

        while len(self) > self.max_size:
            self.observations = self.observation_encoder.getitem(self.observations, slice(1, None))
            self.actions = self.action_encoder.getitem(self.actions, slice(1, None))
            self.rewards = self.rewards[1:]
            self.dones = self.dones[1:]

    def valid_indices(self):
        return torch.where(~self.dones[:-1])[0]

    def sample(self, batch_size: int):
        indices = self.valid_indices()
        indices = indices[torch.randperm(len(indices))]
        indices = indices[:batch_size]

        return self[indices]

    def to(self, device):
        self.observations = self.observations.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)

        return self

    def __len__(self):
        return len(self.valid_indices())

    def __getitem__(self, item):
        # convert sclices to indices
        next_item = torch.arange(1, len(self.dones))[item]

        return (
            self.observation_encoder.getitem(self.observations, item),
            self.action_encoder.getitem(self.actions, next_item),
            self.rewards[next_item],
            self.dones[next_item],
            self.observation_encoder.getitem(self.observations, next_item),
        )
