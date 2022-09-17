from typing import Optional

import os
from pathlib import Path

import gym

from general_q.agents import Agent, GeneralQ
from general_q.utils import play

SAVE_PATH = Path("tmp/pretrained/")


def last_saved_model(path: str) -> Optional[str]:
    path = Path(path)
    files = {
        path: path.stat().st_mtime
        for path in path.iterdir()
    }

    return max(files, key=files.get, default=None)


def train(episodes=2000) -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    # env = gym.wrappers.TransformReward(
    #     gym.make("LunarLander-v2", render_mode="human"), lambda r: 0.1 * r
    # )
    # env = gym.make("Acrobot-v1", render_mode="human")
    # env = gym.wrappers.TransformReward(gym.make('Pendulum-v1', render_mode="human"), lambda r: 0.1*r)
    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # env = gym.make('CliffWalking-v0')
    # env = gym.make('CliffWalking-v0', render_mode="human")
    # env = gym.make("CarRacing-v2", render_mode="human")
    # env = gym.wrappers.TimeLimit(gym.make("CarRacing-v2", render_mode="human"), max_episode_steps=100)
    # env = gym.make("BipedalWalker-v3", render_mode="human")
    # env = gym.make("Blackjack-v1", render_mode="human")

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    path = last_saved_model(SAVE_PATH)
    agent: Agent = \
        path and GeneralQ.load_pretrained(path, raise_error=False) or \
        GeneralQ(
            env.action_space,
            env.observation_space,
            n_samples=8,
        )

    print("Training:")
    print(f"\tAgent: {agent}")
    print(f"\tEnvironment: {env}")

    with env, agent:
        play(env, agent, 1000, train=True)
        agent.save_pretrained(SAVE_PATH)


if __name__ == "__main__":
    train()
