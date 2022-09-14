import gym
import wandb

from gym.wrappers import *
from tqdm import trange
from agents.gq import QAgent
from utils import episode


AGENT_NAME = "Mira-Cart"
AGENT_CLASS = QAgent
SAVE_PATH = f"agents/pretrained/{AGENT_NAME}.{AGENT_CLASS.__name__}"


def train(episodes=2000) -> None:
    wandb.init(project="gq-agents")

    env = gym.make("CartPole-v1", render_mode="human")
    # env = gym.make("Acrobot-v1", render_mode="human")
    # env = TransformReward(gym.make('Pendulum-v1', render_mode="human"), lambda r: 0.1*r)
    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # env = TransformReward(gym.make("LunarLander-v2", render_mode="human"), lambda r: 0.1*r)
    # env = gym.make("CarRacing-v2", render_mode="human")
    # env = TimeLimit(gym.make("CarRacing-v2", render_mode="human"), max_episode_steps=100)
    # env = gym.make("BipedalWalker-v3", render_mode="human")
    # env = gym.make("Blackjack-v1", render_mode="human")

    agent = (
        # QAgent.load_pretrained(SAVE_PATH, raise_error=False) or
        QAgent(
            env.action_space,
            env.observation_space,
            AGENT_NAME,
            n_samples=16,
        )
    )

    with env, agent, trange(episodes) as pbar:
        for episode_num in pbar:
            steps, reward, mean_loss = episode(env, agent, train=True)

            pbar.set_description(
                f"{steps=:3d}, "
                f"{reward=:6.3f}, "
                f"{mean_loss=:6.3f}"
            )

            wandb.log({
                "steps": steps,
                "reward": reward,
                "loss": mean_loss,
            })

            agent.save_pretrained(SAVE_PATH)


if __name__ == "__main__":
    train()
