from typing import Optional

import math
import random

import torch
from gymnasium import Space
from gymnasium.core import ActType, ObsType
from torch import nn, optim
from torch.nn import functional as F

from general_q.agents.base import Agent
from general_q.agents.replay_memory import ReplayMemory
from general_q.encoders import DiscreteEncoder, Encoder, auto_encoder


class GeneralQ(Agent, nn.Module):
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
            replay_memory_size: int = 2**16,
    ) -> None:
        """
        Args:
            action_space: The action space.
            observation_space: The observation space.
            name: The name of the agent. If None, it will be chosen randomly from general_q.agents.NAMES
            action_encoder: The action encoder. Should implement `.all()` method.
            observation_encoder: The observation encoder. Defaults to {auto_encoder} which will choose the best encoder for the observation space.
            q: The Q model, maps from inp: [batch_size, embed_dim] to values: [batch_size, 1]
            embed_dim: The embedding dimension.
            device: The device to use.
            lr: Learning rate
            epsilon: Exploration rate, updated in `agent.update_epsilon()` which is called every time in `agent.learn()` 
            epsilon_decay: `epsilon <- epsilon * sigmoid(-epsilon_decay)`
            gamma: Discount factor, `q(s, a) <- r + sigmoid(gamma) * max(a', q(s', a'))`
            replay_memory_size: The size of the replay memory.
        """

        super().__init__(action_space, observation_space, name)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.q_model = q_model
        if self.q_model is None:
            self.q_model = nn.Sequential(
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

        # TODO move encoders to the q module to add more flexibility and
        # TODO manage unintended side effects of changing encoders and q model
        # TODO after initialization
        self.action_encoder = action_encoder(action_space, embed_dim)
        assert isinstance(self.action_encoder, Encoder), \
            f"{self.action_encoder} should inherit from {Encoder}"

        self.observation_encoder = observation_encoder(observation_space, embed_dim)
        assert isinstance(self.observation_encoder, Encoder), \
            f"{self.observation_encoder} should inherit from {Encoder}"

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )

        self.to(device)

        self.gameplays = ReplayMemory(
            action_encoder=self.action_encoder,
            observation_encoder=self.observation_encoder,
            capacity=replay_memory_size,
        )

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
        self.gameplays.append(observation)

    def remember_transition(
            self,
            new_observation,
            action,
            reward,
            termination,
            truncation,
    ) -> None:
        self.gameplays.append(
            new_observation,
            action,
            reward,
            termination,
            truncation,
        )

    def learn(self, batch_size=128) -> float:
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

        (
            observations,
            actions,
            rewards,
            terminations,
            truncations,
            next_observations,
        ) = self.gameplays.sample(batch_size)

        loss = self.calculate_loss(
            observations,
            actions,
            rewards,
            terminations,
            truncations,
            next_observations,
        )

        # idx = torch.where(self.gameplays.terminations[:-1] & ~self.gameplays.terminations[1:])[0][0]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def calculate_loss(
            self,
            observations_,
            actions_,
            rewards,
            terminations,
            truncations,
            next_observations_,
    ) -> torch.Tensor:
        """
        Calculate the loss value for this specified transitions
        """
        # fmt: off
        observations         = self.observation_encoder(observations_)       # [batch_size, s, emb]
        actions              = self.action_encoder(actions_)                 # [batch_size, s, emb]
        next_observations    = self.observation_encoder(next_observations_)  # [batch_size, s, emb]
        _, action_embeddings = self.action_encoder.all()                     # [n, s, emb]

        next_observations = next_observations.sum(dim=-2, keepdim=True)          # [batch_size, 1, emb]
        action_embeddings = action_embeddings.sum(dim=-2)                        # [n, emb]
        next_qs           = self.q_model(next_observations + action_embeddings)  # [batch_size, n]
        best_q, _         = next_qs.max(dim=-1)                                  # [batch_size]
        # discount factor is zero where the episode has terminated and 
        # sigmoid(self.gamma) everywhere else
        discount_factor   = ~terminations / (1 + math.exp(-self.gamma))          # [batch_size]
        q_target          = rewards + best_q * discount_factor                   # [batch_size]

        observations = observations.sum(dim=-2)              # [batch_size, emb]
        actions      = actions.sum(dim=-2)                   # [batch_size, emb]
        q            = self.q_model(observations + actions)  # [batch_size]
        loss         = F.mse_loss(q, q_target.detach())
        # fmt: on

        return loss

    def update_epsilon(self):
        self.epsilon /= 1 + math.exp(self.epsilon_decay)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def reset(self):
        self.gameplays.close()
