import numpy as np


class ReplayBuffer:
    def __init__(self, state_shape: tuple, action_shape: tuple, max_size: int):
        self.states = np.zeros((0,) + state_shape, dtype=np.float32)
        self.actions = np.zeros((0,) + action_shape, dtype=np.int32)
        self.rewards = np.zeros(0, dtype=np.float32)
        self.dones = np.zeros(0, dtype=bool)

        self.max_size = max_size

    def append(self, state, action, reward, done):
        self.states = np.append(self.states, state[np.newaxis], axis=0)
        self.actions = np.append(self.actions, action[np.newaxis], axis=0)
        self.rewards = np.append(self.rewards, [reward])
        self.dones = np.append(self.dones, [done])

        if len(self) > self.max_size:
            self.states = self.states[-self.max_size:]
            self.actions = self.actions[-self.max_size:]
            self.rewards = self.rewards[-self.max_size:]
            self.dones = self.dones[-self.max_size:]

    def sample(self, batch_size: int):
        return self[np.random.randint(0, len(self), batch_size)]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, item):
        return self.states[item], self.actions[item], self.rewards[item], self.dones[item]
