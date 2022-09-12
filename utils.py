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
    current_loss = 0
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
            # render on multiples of render_step
            if step % render_step == 0 and 'human' in env.metadata['render_modes']:
                env.render(mode='human')

            description = ""
            action = agent(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if train:
                agent.remember(obs, action, reward, done)
                current_loss = agent.learn()

                total_loss += current_loss

            description += f"{current_loss=:6.3f}, {reward=:6.3f}, {total_reward=:6.3f}"
            pbar.set_description(description)

            if done:
                break

    return step, total_reward, total_loss / step
