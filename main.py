import gym
import wandb

from gym.wrappers import *
from tqdm import trange
from agents.gq import QAgent
from utils import episode


AGENT_NAME = "Mira"
AGENT_CLASS = QAgent
SAVE_PATH = f"agents/pretrained/{AGENT_NAME}.{AGENT_CLASS.__name__}"


def train(episodes=2000) -> None:
    wandb.init(project="gq-agents")

    # env = gym.make("CartPole-v1")
    # env = gym.make("Acrobot-v1")
    # env = TransformReward(gym.make('Pendulum-v1'), lambda r: 0.1*r)
    # env = gym.make("MountainCarContinuous-v0")
    # env = TransformReward(gym.make("LunarLander-v2"), lambda r: 0.1*r)
    # env = gym.make("BipedalWalker-v3")
    env = gym.make("Blackjack-v1")

    with env:
        agent = (
            QAgent.load_pretrained(SAVE_PATH, raise_error=False) or
            QAgent(
                env.action_space,
                env.observation_space,
                AGENT_NAME,
                n_samples=20,
            )
        )

        with trange(episodes) as pbar:
            for episode_num in pbar:
                steps, reward, mean_loss = episode(env, agent, render_step=1, train=True)

                pbar.set_description(
                    f"{steps=:3d}, "
                    f"{reward=:.3f}, "
                    f"{mean_loss=:.3f}"
                )

                wandb.log({
                    "steps": steps,
                    "reward": reward,
                    "loss": mean_loss,
                })

                agent.save_pretrained(SAVE_PATH)


if __name__ == "__main__":
    train()
