import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union
from agents.agent import AgentBase
from agents.replay_buffer import ReplayBuffer
from gym.spaces import Box, Discrete, Space
from torch.nn import functional as F
from torch import nn, optim


class QModel(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_encoder = nn.Linear(4, embed_dim)
        self.action_encoder = nn.Embedding(2, embed_dim)
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

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute q values for each action.

        Args:
            obs: [batch_size, 4]
            actions: [batch_size, m]

        Returns:
            qs: [batch_size, m]
        """
        state_embedding = self.observation_encoder(obs).unsqueeze(1)  # [batch_size, 1, embed_dim]
        action_embedding = self.action_encoder(actions)  # [batch_size, m, embed_dim]
        embedding = state_embedding + action_embedding  # [batch_size, m, embed_dim]
        qs = self.decoder(embedding)  # [batch_size, m, 1]
        return qs.squeeze(2)  # [batch_size, m]

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
        gamma: float = 0.99,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.0,
        replay_max_size: int = 1024,
    ) -> None:
        super().__init__(action_space, observation_space, name)

        assert (
            isinstance(action_space, Discrete) and action_space.n == 2
        ), "This agent only works with Discrete(2) action spaces"

        assert isinstance(observation_space, Box) and observation_space.shape == (
            4,
        ), "This agent only works with Box(shape=(4,)) observation spaces"

        self.gameplays = ReplayBuffer(
            state_shape=(4,),
            action_shape=(),
            max_size=replay_max_size,
        )

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
            lr=1e-6,
        )

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def act(self, obs: np.ndarray) -> int:
        """Choose an action for the given observation"""

        if np.random.random() < self.epsilon:
            action = self.action_space.sample()
        else:
            obs = torch.from_numpy(obs).to(self.device, dtype=torch.float).unsqueeze(0)
            actions = torch.tensor([0, 1], dtype=torch.long, device=self.device).unsqueeze(0)
            q = self.model(obs, actions).squeeze(0)
            action = q.argmax().item()

        return action

    def reward(self, reward: float) -> None:
        """Give a reward for last action remembered by the agent"""
        self.gameplays.rewards[-1] += reward

    def remember(self, new_obs, action=0, reward=0, done=False):
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
        action = np.array(action)
        self.gameplays.append(new_obs, action, reward, done)

    def learn(self, batch_size=512) -> float:
        """
        Learn from the gameplays.

        Args:
            batch_size: The number of gameplays to learn from.

        Returns:
            The loss value.
        """
        if len(self.gameplays) < batch_size:
            return 0.0

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        states, actions, rewards, dones, next_states = self.gameplays.sample(batch_size)
        states = torch.from_numpy(states).to(self.device, dtype=torch.float32)
        actions = torch.from_numpy(actions).to(self.device, dtype=torch.long).unsqueeze(1)
        rewards = torch.from_numpy(rewards).to(self.device, dtype=torch.float32)
        dones = torch.from_numpy(dones).to(self.device, dtype=torch.bool)
        next_states = torch.from_numpy(next_states).to(self.device, dtype=torch.float32)
        all_actions = torch.tensor(
            [[0, 1]],
            device=self.device,
            dtype=torch.long,
        ).repeat(batch_size, 1)

        next_qs = self.model(next_states, all_actions)  # [batch_size, 2]
        best_q = next_qs.max(1).values.detach()  # [batch_size]

        q_target = best_q * ~dones * self.gamma + rewards  # [batch_size]
        qs = self.model(states, actions).squeeze(1)  # [batch_size, 1]
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
            The loaded model.
        """
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            if raise_error:
                raise
