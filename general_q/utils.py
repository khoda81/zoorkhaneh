from typing import Optional, Union

import math
import pickle
import time
from collections.abc import Mapping
from itertools import count
from pathlib import Path

from gymnasium import Env, ObservationWrapper, spaces
from tqdm import tqdm, trange

from general_q.agents import Agent

__all__ = ["WallTimeObserver", "evaluate", "load_pretrained", "save_pretrained"]


def evaluate(
    env: Env,
    agent: Agent,
    steps: int,
    train: bool = True,
    step_callback: callable = None,
    episode_callback: callable = None,
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
        episode_callback = lambda **_: None

    if step_callback is None:
        step_callback = lambda **_: None

    try:
        max_step = env.spec.max_episode_steps
    except AttributeError:
        max_step = None

    with trange(steps) as step_pbar:
        while True:
            log_info = {}

            observation, log_info["env"] = env.reset()
            agent.reset()

            step_callback(
                observation=observation,
                log_info=log_info,
                step=step_pbar.n,
            )

            if train:
                agent.remember_initial(observation)

            episode_reward = 0
            with tqdm(count(1), leave=False, total=max_step) as episode_pbar:
                for episode_step in episode_pbar:
                    action = agent(observation)
                    log_info = {}
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        log_info["env"],
                    ) = env.step(action)

                    episode_reward += reward

                    description = ""
                    if train:
                        agent.remember_transition(
                            action,
                            reward,
                            terminated,
                            truncated,
                            observation,
                        )
                        log_info["train"] = agent.learn()

                        if "loss" in log_info["train"]:
                            loss = float(log_info["train"]["loss"])
                            description += f"{loss=:7.3f}, "

                    step_callback(
                        action=action,
                        observation=observation,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        log_info=log_info,
                        step=step_pbar.n,
                    )

                    description += f"{reward=:6.2f}, {episode_reward=:6.2f}"
                    episode_pbar.set_description(description)
                    step_pbar.update()

                    if step_pbar.n >= steps:
                        return

                    if terminated or truncated:
                        break

            episode_callback(
                length=episode_step,
                episode_reward=episode_reward,
                step=step_pbar.n,
            )

            step_pbar.set_description(f"{episode_reward=:6.2f}")


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
        raise TypeError(
            f"Invalid type for 'path': expected str or Path, got {type(path).__name__}"
        )

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
        raise TypeError(
            f"Invalid type for 'path': expected str or Path, got {type(path).__name__}"
        )

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        if raise_error:
            raise pickle.UnpicklingError(
                f"Error loading agent from {path}") from e


class WallTimeObserver(ObservationWrapper):
    """
    An observation wrapper that adds wall-time to the observation.
    """

    def __init__(self, env: Env, base=256):
        super().__init__(env)
        self.base = base
        self.n = math.ceil(math.log(time.time(), base)) + 1
        self.observation_space = spaces.Dict(
            observation=env.observation_space,
            time=spaces.Box(low=0, high=1, shape=(self.n,), dtype=float),
        )

    def observation(self, observation) -> dict:
        remainder = time.time()
        time_obs = []
        for _ in range(self.n):
            time_obs.append(remainder % 1)
            remainder /= self.base

        return {
            "observation": observation,
            "time": time_obs,
        }


def flatten_dict(d: Mapping, parent_key: tuple = ()):
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, Mapping):
            yield from flatten_dict(v, new_key)
        else:
            yield new_key, v
