import torch
from gymnasium import Space, spaces

from general_q.encoders import DictEncoder, auto_encoder


class InvalidMemoryState(Exception):
    pass


class ReplayMemory:
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            capacity: int,
            auto_truncate: bool = True,
            device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
    ):
        # TODO stop allocating all memory at once, implement dynamic list
        self.observation_space = observation_space
        self.action_space = action_space

        space = spaces.Dict(
            action=action_space,
            reward=spaces.Box(low=-float("inf"), high=float("inf"), shape=()),
            terminated=spaces.Box(low=0, high=1, shape=(), dtype=bool),
            truncated=spaces.Box(low=0, high=1, shape=(), dtype=bool),
            new_observation=observation_space,
        )

        self.encoder = DictEncoder(space, subencoder=auto_encoder, embed_dim=None).to(device)
        self.storage = self.encoder.sample([capacity])
        self.storage.map["truncated"].data[:] = True

        self.auto_truncate = auto_truncate
        self.capacity      = capacity
        self.last          = capacity - 1
        self.size          = 0

    def to(self, device: torch.device):
        self.storage = self.storage.transformed(lambda tensor: tensor.to(device))
        self.encoder.to(device)

    def append_initial(self, new_observation) -> None:
        state = self.encoder.prepare({
            "reward"         : 0.,
            "terminated"     : False,
            "truncated"      : False,
            "new_observation": new_observation,
        })

        if self.auto_truncate:
            self.storage.map["truncated"].data[self.last] = True
        else:
            term = self.storage.map["terminated"].data[self.last]
            trunc = self.storage.map["truncated"].data[self.last]

            if not (term or trunc):
                raise InvalidMemoryState(
                    f"No action was provided meaning this is the initial observation, but "
                    f"the last memory state is {~term * 'non-'}terminal and {~trunc * 'non-'}truncated. "
                )

        self.last = (self.last + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.storage[self.last] = state

    def append_transition(
            self,
            action,
            reward,
            terminated,
            truncated,
            new_observation,
    ) -> None:
        transition = self.encoder.prepare({
            "action"         : action,
            "reward"         : reward,
            "terminated"     : terminated,
            "truncated"      : truncated,
            "new_observation": new_observation,
        })

        self.last = (self.last + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.storage[self.last] = transition

    def __bool__(self):
        return self.size > 0

    def __len__(self):
        return self.size
