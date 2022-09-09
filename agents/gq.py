import numpy as np

from torch import nn, distributions as D

from agents.agent import AgentBase
from agents.replay_buffer import ReplayBuffer


class Model(nn.Module):
    def __init__(
        self,
        embed_dim=64,
        action_size=2,
        n=8,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.n = n
        self.q_mean = nn.Linear(embed_dim, n)
        self.q_scale = nn.Linear(embed_dim, n)
        self.q_coef = nn.Linear(embed_dim, n)

        self.action_embedding = nn.Embedding(action_size, embed_dim)
        self.embedding_encoder = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Relu())

    def forward(self, obs, actions):
        """
        obs: [batch_size, 4]
        actions: [batch_size, m, 1]
        """

        state_embedding = self.encoder(obs).unsqueeze(1)    # [batch_size, 1, embed_dim]
        action_embedding = self.action_embedding(actions)   # [batch_size, m, embed_dim]

        embedding = state_embedding + action_embedding  # [batch_size, m, embed_dim]
        encoded = self.embedding_encoder(embedding)     # [batch_size, m, embed_dim]

        logits = self.q_coef(encoded)   # [batch_size, m, n]
        mean = self.q_mean(encoded)     # [batch_size, m, n]
        scale = self.q_scale(encoded)   # [batch_size, m, n]

        mix = D.Categorical(logits=logits)      # [batch_size, m, n]
        comp = D.Normal(mean, scale)            # [batch_size, m, n]
        dist = D.MixtureSameFamily(mix, comp)   # [batch_size, m]

        return dist


class QAgent(AgentBase):
    def __init__(
        self,
        name,
        model=None,
        gamma=0.99,
        max_gameplays=10,
    ):
        super().__init__(name)

        self.gameplays = ReplayBuffer(
            state_shape=(4,),
            action_shape=(1,),
            max_size=200,
        )

        model = Model() if model is None else model

        self.model = model
        self.gamma = gamma
        self.max_gameplays = max_gameplays

    def act(self, state: tuple[np.ndarray, bool], remember=False) -> int:
        raise NotImplementedError

    def reward(self, reward: float) -> None:
        """Give a reward for last action remembered by the agent"""
        self.gameplays.rewards[-1] += reward

    def learn(self, batch_size=64) -> None:
        """Train on all gameplays and update q-model"""
        raise NotImplementedError

    def remember(self, state, action, reward, done) -> None:
        self.gameplays.append(state, action, reward, done)
