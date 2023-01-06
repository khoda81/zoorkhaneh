from typing import Optional, Union

import pickle
from itertools import count
from pathlib import Path

from gymnasium import Env
from tqdm import tqdm, trange

from general_q.agents import Agent


def evaluate(
        env: Env,
        agent: Agent,
        steps: int,
        train: bool = True,
        step_callback: callable = None,
        episode_callback: callable = None
) -> None:
    """
    Run a single episode of the environment.

    Args:
        env: The environment to run the episode in.
        agent: The agent to train.
        steps: The number of steps to run the environment for.
        train: Whether to train the agent.
        step_callback: A callback to call after each step.
        episode_callback: A callback to call after each episode.
    """
    if episode_callback is None:
        episode_callback = lambda *_, **__: None

    if step_callback is None:
        step_callback = lambda *_, **__: None

    try:
        max_step = env.spec.max_episode_steps
    except AttributeError:
        max_step = None

    with trange(steps) as step_pbar:
        while True:
            observation, info = env.reset()
            agent.reset()

            if train:
                agent.remember_initial(observation)

            acc_rew = 0
            eps_loss = 0
            with tqdm(count(1), leave=False, total=max_step) as episode_pbar:
                for episode_step in episode_pbar:
                    action = agent(observation)
                    observation, reward, terminated, truncated, info = env.step(action)

                    step_callback(
                        observation=observation,
                        action=action,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        step=step_pbar.n,
                    )

                    acc_rew += reward

                    description = ""
                    if train:
                        agent.remember_transition(observation, action, reward, terminated, truncated)

                        loss = agent.learn()
                        description += f"{loss=:7.3f}, "
                        eps_loss += loss

                    description += f"{reward=:6.2f}, {acc_rew=:6.2f}"
                    episode_pbar.set_description(description)
                    step_pbar.update()

                    if step_pbar.n >= steps:
                        return

                    if terminated or truncated:
                        break

            episode_callback(
                step=step_pbar.n,
                length=episode_step,
                loss=eps_loss / episode_step,
                reward=acc_rew,
            )

            step_pbar.set_description(f"last_acc_reward={acc_rew:6.2f}")


def save_pretrained(
    agent: Agent,
    path: Union[str, Path],
    raise_error: bool = True,
):
    """
    Save the agent to the given path.

    Args:
        path: The path to save the agent to.
    """

    if not isinstance(path, (str, Path)):
        raise TypeError(f"Invalid type for 'path': expected str or Path, got {type(path).__name__}")

    path = Path(path) / f"{agent.name}.{agent.__class__.__name__}"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "wb") as f:
            pickle.dump(agent, f)
    except (OSError, AttributeError, pickle.PicklingError) as e:
        if raise_error:
            raise pickle.PicklingError(f"Error saving {agent} to {path}") from e


def load_pretrained(
        path: Union[str, Path],
        raise_error: bool = True,
) -> Optional[Agent]:
    """
    Try to load the agent from the given path.

    Args:
        path: The path to load the agent from.
        raise_error: Whether to raise an error if the agent could not be loaded.

    Returns:
        The loaded agent if the agent could be loaded.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Invalid type for 'path': expected str or Path, got {type(path).__name__}")

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        if raise_error:
            raise pickle.UnpicklingError(f"Error loading agent from {path}") from e
