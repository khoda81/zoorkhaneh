from itertools import count

from gym import Env
from tqdm import tqdm

from general_q.agents import Agent


def episode(env: Env, agent: Agent, train=True) -> tuple[int, float, float]:
    """
    Run a single episode of the environment.

    Args:
        env: The environment to run the episode in.
        agent: The agent to use to interact with the environment.
        train: Whether to train the agent.

    Returns:
        A tuple containing the number of steps taken, the total reward, and the
        average loss.
    """
    observation, info = env.reset()
    agent.reset()

    if train:
        # remember with no action, reward, done
        agent.remember(observation)

    tot_rew = 0
    loss = 0
    total_loss = 0
    try:
        max_step = env.spec.max_episode_steps
    except AttributeError:
        max_step = None

    # count() is a generator that counts up
    # leave=False means that the progress bar will be cleared after each episode
    # useful for running multiple episodes in a loop with tqdm progress bar
    with tqdm(count(1), leave=False, total=max_step) as pbar:
        for step in pbar:
            description = ""
            action, value = agent(observation)
            observation, reward, termination, truncation, info = env.step(
                action
            )
            tot_rew += reward

            if train:
                agent.remember(
                    observation, action, reward, termination, truncation
                )
                loss = agent.learn()

                total_loss += loss

            pbar.set_description(
                f"{loss=:6.3f}, {reward=:6.2f}, {tot_rew=:6.2f}, {value=:6.2f}"
            )

            if termination or truncation:
                break

    return step, tot_rew, total_loss / step
