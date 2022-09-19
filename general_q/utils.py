from itertools import count

from gym import Env
from tqdm import tqdm, trange

from general_q.agents import Agent


def play(
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
        agent: The agent to use to interact with the environment.
        steps: The number of steps to run the environment for.
        train: Whether to train the agent.
    """
    if episode_callback is None:
        episode_callback = lambda *args: None
    if step_callback is None:
        step_callback = lambda *args: None

    try:
        max_step = env.spec.max_episode_steps
    except AttributeError:
        max_step = None

    with trange(steps) as step_pbar:
        while True:
            observation, info = env.reset()
            agent.reset()

            if train:
                agent.remember(observation)

            tot_rew = 0
            eps_loss = 0
            with tqdm(count(1), leave=False, total=max_step) as episode_pbar:
                for episode_step in episode_pbar:
                    action, value = agent(observation)
                    observation, reward, termination, truncation, info = env.step(action)
                    tot_rew += reward

                    step_callback(step_pbar.n, observation, action, reward, termination, truncation)

                    description = ""
                    if train:
                        agent.remember(observation, action, reward, termination, truncation)
                        loss = agent.learn()
                        description += f"{loss=:6.3f}, "
                        eps_loss += loss

                    description += f"{reward=:6.2f}, {tot_rew=:6.2f}, {value=:6.2f}"
                    episode_pbar.set_description(description)
                    step_pbar.update()

                    if step_pbar.n >= steps:
                        return

                    if termination or truncation:
                        break

            episode_callback(
                step_pbar.n, 
                episode_step, 
                eps_loss / episode_step, 
                tot_rew
            )
            step_pbar.set_description(f"last_reward: {tot_rew:6.2f}")
