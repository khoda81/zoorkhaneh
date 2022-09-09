from itertools import count
from tqdm import tqdm


def episode(env, agent, render_step=10, train=True) -> tuple[int, float, float]:
    """Run a single episode of the environment.

    Args:
        env: The environment to run the episode in.
        agent: The agent to use to interact with the environment.
        render_step: The step interval at which to render the environment.
        train: Whether to train the agent.

    Returns:
        A tuple containing the number of steps taken, the total reward, and the
        average loss.
    """
    obs = env.reset()
    agent.reset()

    if train:
        # remember with no action, reward, done
        agent.remember(obs)

    total_reward = 0
    total_loss = 0

    # count() is a generator that counts up
    # leave=False means that the progress bar will be cleared after each episode
    # useful for running multiple episodes in a loop with tqdm progress bar
    try:
        max_step = env.spec.max_episode_steps
    except AttributeError:
        max_step = None

    for step in tqdm(count(), leave=False, total=max_step):
        # render on multiples of render_step
        if (step + 1) % render_step == 0:
            env.render()

        action = agent(obs)
        obs, reward, done, info = env.step(action)

        if train:
            # remember this transition for training
            agent.remember(obs, action, reward, done)
            total_loss += agent.learn()

        total_reward += reward

        if done:
            break

    steps = step + 1
    return steps, total_reward, total_loss / steps
