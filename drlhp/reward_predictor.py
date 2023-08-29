from collections import deque
from dataclasses import dataclass
from queue import Empty
import random
import torch
from drlhp.models import CNN, MLP

from drlhp.config import PPOConfig
from multiprocessing import Queue, Event


@dataclass
class PreferenceExample:
    obs_actions1: list[tuple[torch.Tensor, int]]
    obs_actions2: list[tuple[torch.Tensor, int]]
    env_rewards1: list[float]
    env_rewards2: list[float]
    mu: float


class PreferenceDatabase:
    def __init__(self):
        self._examples = deque(maxlen=1000)

    def add_example(self, example: PreferenceExample):
        self._examples.append(example)


def return_ones(x: torch.Tensor) -> torch.Tensor:
    n_envs = x.shape[0]
    return torch.ones(n_envs, device=x.device, dtype=torch.float32)


class RewardFunction:
    def __init__(self, observation_size: int, action_space: int, config: PPOConfig, env_name: str):
        self.config = config
        self.env_name = env_name
        self.action_space_size = action_space
        self.observation_space_size = observation_size
        self.gamma = config.gamma
        self.tau = config.tau

        if "CartPole" in env_name:
            self.layer_sizes = (
                [observation_size + action_space] + config.critic_n_layers * [config.critic_hidden_dim] + [1]
            )
            self.model = MLP(self.layer_sizes)
        else:
            self.model = CNN(1)

    def train_on_oracle(self, preference_database: PreferenceDatabase):
        breakpoint()
        data = preference_database._examples
        random.shuffle(data)

        # batch the data

        for example in data:
            pass

    def train_on_prefernces():
        pass


def reward_interface_loop(input_queue: Queue, should_exit: Event):  # type: ignore
    # Start the Flask app and pass the input_queue
    PreferenceDatabase()
    print("Reward model started")

    while not should_exit.is_set():
        # check if there is a new observation
        try:
            input_queue.get(timeout=1)
            print("Reward model recieved scored obser ation received observation")

            # clip1, clip2 = segment_database.get_trajectories_for_preference()
            # play_frames_as_video([obs.rgb for obs in clip1])

        except Empty:
            pass
