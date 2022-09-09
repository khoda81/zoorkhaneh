import numpy as np


class ReplayBuffer:
    def __init__(self, state_shape: tuple, action_shape: tuple, max_size: int):
        self.states = np.empty((0,) + state_shape, dtype=np.float32)
        self.actions = np.empty((0,) + action_shape, dtype=np.int32)
        self.rewards = np.empty(0, dtype=np.float32)
        self.dones = np.empty(0, dtype=bool)

        self.max_size = max_size

    def append(self, state, action, reward, done):
        self.states = np.append(self.states, [state], axis=0)
        self.actions = np.append(self.actions, [action], axis=0)
        self.rewards = np.append(self.rewards, [reward])
        self.dones = np.append(self.dones, [done])

        while len(self) > self.max_size:
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.rewards = self.rewards[1:]
            self.dones = self.dones[1:]

    def valid_indices(self):
        return np.where(~self.dones[:-1])[0]

    def sample(self, batch_size: int):
        indices = np.random.choice(
            self.valid_indices(),
            batch_size,
            replace=False,
        )

        return self[indices]

    def __len__(self):
        return len(self.valid_indices())

    def __getitem__(self, item):
        return (
            self.states[item],
            self.actions[1:][item],
            self.rewards[1:][item],
            self.dones[1:][item],
            self.states[1:][item]
        )
