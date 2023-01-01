from general_q.agents.base import NAMES, Agent
from general_q.agents.general_q import GeneralQ
from general_q.agents.replay_memory import ReplayMemory

# TODO add api to make it easier to record human gameplays and train agents on them
# TODO maybe a GeneralAdam to record human gameplays

__all__ = [
    'NAMES',
    'Agent',
    'GeneralQ',
    'ReplayMemory',
]
