from typing import Optional

import math
import random
from itertools import chain

import torch
from gymnasium import Space
from gymnasium.core import ActType, ObsType
from torch import nn, optim
from torch.nn import functional as F

from general_q.agents.base import Agent
from general_q.agents.replay_memory import ReplayMemory
from general_q.encoders import DiscreteEncoder, Encoder, auto_encoder
from general_q.encoders.storage import MapStorage


class DQN(Agent):
    def __init__(
            self,
            action_space: Space[ActType],
            observation_space: Space[ObsType],
            name: Optional[str] = None,
            action_encoder=DiscreteEncoder,
            observation_encoder=auto_encoder,
            q_model: Optional[nn.Module] = None,
            embed_dim: int = 256,
            device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
            lr: float = 1e-4,
            epsilon: float = 5e-1,
            epsilon_decay: float = -7.0,
            gamma: float = 4.0,
            memory_capacity: int = 2**16,
    ) -> None:
        """
        Args:
            action_space: The action space.
            observation_space: The observation space.
            name: The name of the agent. If None, it will be chosen randomly from general_q.agents.NAMES
            action_encoder: The action encoder. Should implement `.all()` method.
            observation_encoder: The observation encoder. Defaults to {auto_encoder} which will choose the best encoder for the observation space.
            q_model: The Q model, maps from inp: [batch_size, embed_dim] to values: [batch_size]
            embed_dim: The embedding dimension.
            device: The device to use.
            lr: Learning rate
            epsilon: Exploration rate, updated in `agent.update_epsilon()` which is called every time in `agent.learn()`
            epsilon_decay: `epsilon <- epsilon * sigmoid(-epsilon_decay)`
            gamma: Discount factor, `q(s, a) <- r + sigmoid(gamma) * max(a', q(s', a'))`
            memory_capacity: The size of the replay memory.
        """

        super().__init__(action_space, observation_space, name)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        if q_model is None:
            q_model = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
                nn.Flatten(-2),
            )

        self.q_model = q_model

        self.action_encoder = action_encoder(action_space, embed_dim)
        assert isinstance(self.action_encoder, Encoder), \
            f"{self.action_encoder} should inherit from {Encoder}"

        self.observation_encoder = observation_encoder(observation_space, embed_dim)
        assert isinstance(self.observation_encoder, Encoder), \
            f"{self.observation_encoder} should inherit from {Encoder}"

        self.optimizer = optim.Adam(
            params=chain(
                self.q_model.parameters(),
                self.observation_encoder.parameters(),
                self.action_encoder.parameters(),
            ),
            lr=lr,
        )

        self.gameplays = ReplayMemory(
            action_space=self.action_space,
            observation_space=self.observation_space,
            capacity=memory_capacity,
            device=device
        )

        self.to(device)

    @torch.no_grad()
    def act(self, obs: ObsType) -> ActType:
        obs = self.observation_encoder.prepare(obs)
        actions, action_embeddings = self.action_encoder.all()
        action_embeddings = action_embeddings.sum(dim=-2)

        obs_embeddings = self.observation_encoder(obs).sum(dim=-2)
        values = self.q_model(obs_embeddings + action_embeddings)

        if random.random() < self.epsilon:
            action_index = random.randint(0, len(action_embeddings) - 1)
        else:
            action_index = values.argmax(dim=0)

        return self.action_encoder.unprepare(actions[action_index])

    def remember_initial(self, observation: ObsType) -> None:
        self.gameplays.append_initial(observation)

    def remember_transition(self, action, reward, terminated, truncated, new_observation) -> None:
        self.gameplays.append_transition(action, reward, terminated, truncated, new_observation)

    def learn(self, batch_size=128) -> dict[str, torch.Tensor]:
        """
        Learn from the gameplays.

        Args:
            batch_size: The number of gameplay steps to learn from.

        Returns:
            The loss value.
        """
        self.update_epsilon()

        batch_size = min(batch_size, len(self.gameplays))
        if batch_size == 0:
            return 0.0

        sample = self.sample_transitions(batch_size)
        loss = self.calculate_loss(sample)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss}

    def sample_transitions(self, batch_size: int) -> MapStorage:
        """
        Sample a batch of transitions from the replay memory.
        """
        terminal = self.gameplays.storage.map["terminated"].data | self.gameplays.storage.map["truncated"].data
        terminal[self.gameplays.last] = True
        last_indices, = torch.where(~terminal)

        last_indices = last_indices[torch.randperm(len(last_indices))]  # randomize indices
        last_indices = last_indices[:batch_size]

        last_indices = torch.arange(self.gameplays.size)[last_indices]  # convert slices to indices
        indices = (last_indices + 1) % self.gameplays.capacity

        sample = self.gameplays.storage[indices]
        sample.map["observation"] = \
            self.gameplays.storage.map["new_observation"][last_indices]

        return sample

    def calculate_loss(self, sample: MapStorage) -> torch.Tensor:
        """
        Calculate the loss value for this specified transitions
        """
        observation          = sample.map["observation"]                  # [batch_size]
        observation          = self.observation_encoder(observation)      # [batch_size, s, emb]
        action               = sample.map["action"]                       # [batch_size]
        action               = self.action_encoder(action)                # [batch_size, s, emb]
        new_observation      = sample.map["new_observation"]              # [batch_size]
        new_observation      = self.observation_encoder(new_observation)  # [batch_size, s, emb]
        _, action_embeddings = self.action_encoder.all()                  # [n, s, emb]

        new_observation   = new_observation.sum(dim=-2, keepdim=True)          # [batch_size, 1, emb]
        action_embeddings = action_embeddings.sum(dim=-2)                      # [n, emb]
        next_qs           = self.q_model(new_observation + action_embeddings)  # [batch_size, n]
        best_q, _         = next_qs.max(dim=-1)                                # [batch_size]

        # discount factor is zero where the episode has terminated and
        # sigmoid(self.gamma) everywhere else
        terminated      = sample.map["terminated"] .data             # [batch_size]
        rewards         = sample.map["reward"].data                  # [batch_size]
        discount_factor = ~terminated / (1 + math.exp(-self.gamma))  # [batch_size]
        q_target        = rewards + best_q * discount_factor         # [batch_size]

        observation = observation.sum(dim=-2)             # [batch_size, emb]
        action      = action.sum(dim=-2)                  # [batch_size, emb]
        q           = self.q_model(observation + action)  # [batch_size]
        loss        = F.mse_loss(q, q_target.detach())

        return loss

    def update_epsilon(self):
        self.epsilon /= 1 + math.exp(self.epsilon_decay)

    def to(self, device: torch.device):
        self.q_model.to(device)
        self.observation_encoder.to(device)
        self.action_encoder.to(device)
        self.gameplays.to(device)
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)

            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
