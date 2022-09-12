import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union
from agents.agent import AgentBase
from agents.encoder import auto_encoder
from agents.replay_buffer import ReplayDeque
from gym.spaces import Space
from torch.nn import functional as F
from torch import nn, optim


class QModel(nn.Module):
    def __init__(
        self,
        action_space: Space,
        observation_space: Space,
        action_encoder=auto_encoder,
        observation_encoder=auto_encoder,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.action_encoder = action_encoder(action_space, embed_dim)
        self.observation_encoder = observation_encoder(observation_space, embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, obs, actions) -> torch.Tensor:
        """
        Compute q values for each action.

        Args:
            obs: [*batch_shape, *observasion_shape]
            actions: [*batch_shape, m, *action_shape]

        Returns:
            qs: [*batch_shape, m]
        """
        obs_embedding = self.observation_encoder(obs).unsqueeze(-2)  # [*batch_shape, 1, embed_dim]
        action_embedding = self.action_encoder(actions)  # [*batch_shape, m, embed_dim]
        embedding = obs_embedding + action_embedding  # [*batch_shape, m, embed_dim]
        qs = self.decoder(embedding)  # [*batch_shape, m, 1]
        return qs.squeeze(-1)  # [*batch_shape, m]

    def prepare_actions(self, actions) -> torch.Tensor:
        return self.action_encoder.prepare(actions)

    def prepare_observations(self, observations) -> torch.Tensor:
        return self.observation_encoder.prepare(observations)

    def __repr__(self):
        return f"{self.__class__.__name__}(embed_dim={self.observation_encoder.out_features})"


class QAgent(AgentBase):
    def __init__(
        self,
        action_space: Space,
        observation_space: Space,
        name: str,
        model: nn.Module = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        lr: float = 1e-5,
        replay_memory_size: int = 128,
        gamma: float = 0.99,
        n_samples: int = 32,
    ) -> None:
        super().__init__(action_space, observation_space, name)

        self.device = device
        self.model = (
            QModel(
                self.action_space,
                self.observation_space,
            )
            if model is None else model
        )
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

        self.gameplays = ReplayDeque(
            action_encoder=self.model.action_encoder,
            observation_encoder=self.model.observation_encoder,
            max_size=replay_memory_size,
        )

        self.gamma = gamma
        self.n_samples = n_samples

    def act(self, obs: np.ndarray) -> int:
        """Choose an action for the given observation"""

        action_samples = self.model.action_encoder.sample(self.n_samples)
        obs = self.model.observation_encoder.prepare(obs)

        q = self.model(obs, action_samples)
        best_action = q.argmax()
        action = self.model.action_encoder.getitem(action_samples, best_action)
        action = self.model.action_encoder.item(action)

        return action

    def reward(self, reward: float) -> None:
        """Give a reward for last action remembered by the agent"""
        self.gameplays.rewards[-1] += reward

    def remember(self, new_obs, action=None, reward=0, done=False):
        """
        Remember the gameplays.

        Args:
            self: The object itself.
            new_obs: The new observation.
            action: The action.
            reward: The reward.
            done: The done.

        Returns:
            None
        """
        if action is None:
            action = self.action_space.sample()

        self.gameplays.append(new_obs, action, reward, done)

    def learn(self, batch_size=64) -> float:
        """
        Learn from the gameplays.

        Args:
            batch_size: The number of gameplays to learn from.

        Returns:
            The loss value.
        """
        if len(self.gameplays) < batch_size:
            return 0.0

        observations, actions, rewards, dones, next_observations = self.gameplays.sample(batch_size)

        action_samples = self.model.action_encoder.sample(self.n_samples)
        action_samples = self.model.action_encoder.getitem(action_samples, None)
        next_qs = self.model(next_observations, action_samples)  # [batch_size, 2]
        best_q = next_qs.max(1).values.detach()  # [batch_size]

        q_target = best_q * ~dones * self.gamma + rewards  # [batch_size]
        actions = self.model.action_encoder.getitem(actions, (slice(None), None))
        qs = self.model(observations, actions).squeeze(1)  # [batch_size, 1]
        loss = F.mse_loss(qs, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """
        Save the model to the given path.

        Args:
            path: The path to save the model to.

        Returns:
            None
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pretrained(
            cls, path: Union[str, Path],
            raise_error: bool = True) -> Optional["QAgent"]:
        """
        Load the model from the given path.

        Args:
            path: The path to load the model from.

        Returns:
            The loaded model or None if the model could not be loaded.
        """
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            if raise_error:
                raise
