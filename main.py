import gym
from gym import Env
from gym.wrappers import TransformReward
from tqdm import trange
from agents.gq import QAgent
from utils import episode


ENV_NAME = "CartPole-v1"
# ENV_NAME = "BipedalWalker-v3"

AGENT_NAME = "Mira"
AGENT_CLASS = QAgent
SAVE_PATH = f"agents/pretrained/{AGENT_NAME}.{AGENT_CLASS.__name__}"


def train(episodes=500) -> None:
    env = gym.make(ENV_NAME)
    env = TransformReward(env, lambda r: 0.01*r)
    with env:
        agent = (
            QAgent.load_pretrained(SAVE_PATH, raise_error=False) or
            QAgent(
                env.action_space,
                env.observation_space,
                AGENT_NAME,
            )
        )

        with trange(episodes) as pbar:
            for _ in pbar:
                steps, reward, loss = episode(env, agent, render_step=20, train=True)
                pbar.set_description(
                    f"{steps=:3d}, "
                    f"{reward=:.3f}, "
                    f"{loss=:.3f}"
                )

                agent.save_pretrained(SAVE_PATH)


if __name__ == "__main__":
    train()
