from typing import Optional

from pathlib import Path

import gymnasium
import wandb

from general_q.agents import Agent, GeneralQ
from general_q.utils import play

SAVE_PATH = Path("tmp/pretrained/")


def train(wandb_project="general-q") -> None:
    # env = gymnasium.make("CartPole-v1", render_mode="human")
    # env = gymnasium.wrappers.TransformReward(
    #     gymnasium.make("LunarLander-v2", render_mode="human"), lambda r: 0.01 * r
    # )
    # env = gymnasium.make("Acrobot-v1", render_mode="human")
    # env = gymnasium.wrappers.TransformReward(gymnasium.make('Pendulum-v1', render_mode="human"), lambda r: 0.1 * r)
    # env = gymnasium.make("MountainCarContinuous-v0", render_mode="human")
    # env = gymnasium.make('CliffWalking-v0')
    # env = gymnasium.make('CliffWalking-v0', render_mode="human")
    # env = gymnasium.make("CarRacing-v2", render_mode="human")
    # env = gymnasium.wrappers.TimeLimit(gymnasium.make("CarRacing-v2", render_mode="human"), max_episode_steps=100)
    # env = gymnasium.make("BipedalWalker-v3", render_mode="human")
    # env = gymnasium.make("Blackjack-v1", render_mode="human")

    agent = load_agent(SAVE_PATH, env) or create_agent(env)

    print("Training:")
    print(f"\tAgent: {agent}")
    print(f"\tEnvironment: {env}")

    def save_agent(step, *args, **kwargs):
        if step % 1000 == 0:
            agent.save_pretrained(SAVE_PATH)

    def log_to_wandb(step, length, loss, reward, *args, **kwargs):
        wandb.log(
            {
                "step":   step,
                "loss":   loss,
                "reward": reward,
                "length": length,
            }
        )

    wandb.init(
        project=wandb_project,
        dir=SAVE_PATH.parent,
        name=str(agent)
    )

    with env, agent:
        play(
            env,
            agent,
            40000,
            train=True,
            step_callback=save_agent,
            episode_callback=log_to_wandb,
        )

    agent.save_pretrained(SAVE_PATH)


def load_agent(path, env: gymnasium.Env) -> Optional[Agent]:
    path = Path(path)

    if not path.exists():
        return None

    best = -float("inf"), None
    for path in path.iterdir():
        agent = GeneralQ.load_pretrained(path, raise_error=False)
        if agent is None:
            continue

        if (agent.action_space, agent.observation_space) != (env.action_space, env.observation_space):
            continue

        candidate = path.stat().st_mtime, agent
        best = max(best, candidate)

    time, agent = best
    return agent


def create_agent(env: gymnasium.Env) -> Agent:
    return GeneralQ(
        env.action_space,
        env.observation_space,
    )


if __name__ == "__main__":
    train()