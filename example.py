import gym
from tqdm import trange

import wandb
from general_q.agents import GeneralQ
from general_q.utils import episode

AGENT_NAME = "Mira-Cart"
AGENT_CLASS = GeneralQ
SAVE_PATH = f"agents/pretrained/{AGENT_NAME}.{AGENT_CLASS.__name__}"


def train(episodes=2000) -> None:
    wandb.init(project="gq-agents")

    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.wrappers.TransformReward(
        gym.make("LunarLander-v2", render_mode="human"), lambda r: 0.1 * r
    )
    # env = gym.make("Acrobot-v1", render_mode="human")
    # env = gym.wrappers.TransformReward(gym.make('Pendulum-v1', render_mode="human"), lambda r: 0.1*r)
    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # env = gym.make('CliffWalking-v0')
    # env = gym.make('CliffWalking-v0', render_mode="human")
    # env = gym.make("CarRacing-v2", render_mode="human")
    # env = gym.wrappers.TimeLimit(gym.make("CarRacing-v2", render_mode="human"), max_episode_steps=100)
    # env = gym.make("BipedalWalker-v3", render_mode="human")
    # env = gym.make("Blackjack-v1", render_mode="human")

    agent = GeneralQ.load_pretrained(SAVE_PATH, raise_error=False) or GeneralQ(
        env.action_space,
        env.observation_space,
        AGENT_NAME,
        n_samples=32,
    )

    with env, agent, trange(episodes) as pbar:
        for _ in pbar:
            steps, reward, mean_loss = episode(env, agent, train=True)

            pbar.set_description(
                f"{steps=:3d}, " f"{reward=:6.3f}, " f"{mean_loss=:6.3f}"
            )

            wandb.log(
                {
                    "steps": steps,
                    "reward": reward,
                    "loss": mean_loss,
                }
            )

            agent.save_pretrained(SAVE_PATH)


if __name__ == "__main__":
    train()
