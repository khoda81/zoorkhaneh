from itertools import count
import gym
import random
import numpy as np
from tqdm import trange, tqdm


# env = gym.make("BipedalWalker-v3")
env = gym.make("CartPole-v1")


def Random_games():
    # Each of this episode is its own game.
    for episode in trange(500):
        obs = env.reset()
        # this is each frame, up to 500...but we wont make it that far with random.
        done = False
        for step in tqdm(count(0), leave=False):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.

            if (step + 1) % 1 == 0:
                env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be any of one how in list on 4, for example [0 1 0 0]
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            obs, reward, done, info = env.step(action)

            # lets print everything in one line:
            #print(reward, action)
            if done:
                break


Random_games()
