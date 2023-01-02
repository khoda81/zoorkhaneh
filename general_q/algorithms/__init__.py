from general_q.algorithms.base import NAMES, Algorithm
from general_q.algorithms.dqn import DQN
from general_q.algorithms.replay_memory import ReplayMemory

# TODO add api to make it easier to record human gameplays and train agents on them
# TODO maybe a GeneralAdam to record human gameplays

__all__ = [
    'NAMES',
    'Algorithm',
    'DQN',
    'ReplayMemory',
]
