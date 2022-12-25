from typing import Optional

import torch
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Space
from torch import nn, optim
from torch.nn import functional as F

from general_q.agents.base import Agent
from general_q.agents.replay_memory import ReplayMemory
from general_q.encoders import DiscreteEncoder, Encoder, auto_encoder


class InvalidMemoryState(Exception):
    pass



class GeneralQ(Agent, nn.Module):
    def __init__(
            self,
            action_space: Space[ActType],
            observation_space: Space[ObsType],
            name: Optional[str] = None,
            action_encoder=DiscreteEncoder,
            observation_encoder=auto_encoder,
            q: Optional[nn.Module] = None,
            embed_dim: int = 256,
            device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
            lr: float = 1e-3,
            replay_memory_size: int = 4096,
    ) -> None:
        super().__init__(action_space, observation_space, name)

        self.q = q
        if self.q is None:
            self.q = nn.Sequential(
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

        self.action_encoder = action_encoder(action_space, embed_dim)
        assert isinstance(self.action_encoder, Encoder), f"{self.action_encoder} should inherit from {Encoder}"

        self.observation_encoder = observation_encoder(observation_space, embed_dim)
        assert isinstance(self.observation_encoder, Encoder), f"{self.observation_encoder} should inherit from {Encoder}"

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

        self.gameplays = ReplayMemory(
            action_encoder=self.action_encoder,
            observation_encoder=self.observation_encoder,
            capacity=replay_memory_size,
        )

    @torch.no_grad()
    def act(self, obs: ObsType) -> tuple[ActType, float]:
        obs = self.observation_encoder.prepare(obs)
        obs_embeddings = self.observation_encoder(obs)
        actions, action_embeddings = self.action_encoder.all()

        values = self.q(obs_embeddings + action_embeddings)
        best_value, best_action = values.max(dim=0)
        action = actions[best_action].item()

        return action, best_value.item()

    def remember(
            self,
            new_observation,
            action=None,
            reward=0,
            termination=False,
            truncation=False,
    ) -> None:
        if action is None:
            action = self.action_space.sample()
            if self.gameplays.size != 0:
                _, _, _, term, trunc, _ = self.gameplays[self.gameplays.last]
                state = "terminal" if term else "truncated" if trunc else ""

                if state:
                    raise InvalidMemoryState(
                        f"The last memory state was {state}, but no action was provided. "
                        f"Did you forget to call `agent.reset()` before `agent.remember(obs)`?"
                    )

        self.gameplays.push(
            new_observation,
            action,
            reward,
            termination,
            truncation,
        )

    def learn(self, batch_size=512) -> float:
        """
        Learn from the gameplays.

        Args:
            batch_size: The number of gameplay steps to learn from.

        Returns:
            The loss value.
        """

        batch_size = min(batch_size, len(self.gameplays))
        if batch_size == 0:
            return 0.0

        (
            observations,
            actions,
            rewards,
            terminations,
            _truncations,
            next_observations,
        ) = self.gameplays.sample(batch_size)

        next_observations = self.observation_encoder(next_observations)[:, None]  # [batch_size, 1, emb]
        _, action_embeddings = self.action_encoder.all()  # [n, emb]
        next_qs = self.q(next_observations + action_embeddings)  # [batch_size, n]
        best_q = next_qs.max(dim=1).values  # [batch_size]
        q_target = best_q * ~terminations + rewards  # [batch_size]

        observations = self.observation_encoder(observations)  # [batch_size, emb]
        actions = self.action_encoder(actions)  # [batch_size, emb]
        qs = self.q(observations + actions)  # [batch_size]
        loss = F.mse_loss(qs, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def reset(self):
        self.gameplays.close()
