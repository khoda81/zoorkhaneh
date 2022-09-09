import gym
from gym import Env

from itertools import count
from tqdm import trange, tqdm

from agents.agent import AgentBase


def episode(
    env,
    agent,
    render_step=10,
    train=True,

):
    obs = env.reset()
    agent.reset()

    if train:
        # remember with no action, reward, done
        agent.remember(obs)

    total_reward = 0

    # count() is a generator that counts up
    # leave=False means that the progress bar will be cleared after each episode
    # useful for running multiple episodes in a loop with tqdm progress bar
    for step in tqdm(count(), leave=False):
        # render on multiples of render_step
        if (step + 1) % render_step == 0:
            env.render()

        action = agent(obs)
        obs, reward, done, info = env.step(action)

        if train:
            # remember this transition for training
            agent.remember(obs, action, reward, done)
            agent.learn()

        total_reward += reward

        if done:
            break

    return step + 1, total_reward


def random_games() -> None:
    with gym.make(ENV_NAME) as env:
        env: Env

        agent = AgentBase(
            env.action_space,
            env.observation_space,
            "Mira",
        )

        for _ in trange(10):
            episode(env, agent)


if __name__ == "__main__":
    ENV_NAME = "CartPole-v1"
    # ENV_NAME = "BipedalWalker-v3"
    random_games()
