from typing import Optional, Union

import pickle
from pathlib import Path

import torch
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Space
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
            q: Optional[nn.Module] = None,
            embed_dim: int = 256,
            device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
            lr: float = 1e-3,
            replay_memory_size: int = 4096,
    ) -> None:
        Agent.__init__(self, action_space, observation_space, name)
        nn.Module.__init__(self)

        self.device = device
        if q is None:
            q = nn.Sequential(
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

            q.optimizer = optim.Adam(q.parameters(), lr=lr)

        self.q = q

        self.action_encoder = action_encoder(action_space, embed_dim)
        assert isinstance(self.action_encoder, Encoder), f"{self.action_encoder} should inherit from {Encoder}"

        self.observation_encoder = observation_encoder(observation_space, embed_dim)
        assert isinstance(self.observation_encoder, Encoder), f"{self.observation_encoder} should inherit from {Encoder}"

        self.to(self.device)

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
            batch_size: The number of gameplays to learn from.

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

        self.q.optimizer.zero_grad()
        loss.backward()
        self.q.optimizer.step()

        return loss.item()

    def reset(self):
        # set the last transition to invalid
        self.gameplays.close()

    def save_pretrained(self, path: Union[str, Path]):
        """
        Save the model to the given path.

        Args:
            path: The path to save the model to.
        """

        path = Path(path) / f"{self.name}.{self.__class__.__name__}"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pretrained(
            cls,
            path: Union[str, Path],
            raise_error: bool = True,
    ) -> Optional["GeneralQ"]:
        """
        Load the model from the given path.

        Args:
            path: The path to load the model from.
            raise_error: Whether to raise an error if the model could not be loaded.

        Returns:
            The loaded model or None if the model could not be loaded.
        """
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            if raise_error:
                raise
