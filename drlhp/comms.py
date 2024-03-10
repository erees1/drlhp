from dataclasses import dataclass
from multiprocessing import Queue
from typing import Generic, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray
from slist import Slist

T = TypeVar("T")


class TypedQueue(Generic[T]):
    def __init__(self, maxsize: int = 0):
        self.queue = Queue(maxsize=maxsize)

    def put(self, item: T):
        self.queue.put(item)

    def get(self, timeout: float | None = None) -> T:
        return self.queue.get(timeout=timeout)


@dataclass(frozen=True)
class TrajectoryForPreference:
    obs_actions1: list[tuple[torch.Tensor, int]]
    obs_actions2: list[tuple[torch.Tensor, int]]
    env_rewards1: list[float]
    env_rewards2: list[float]


@dataclass(frozen=True)
class Observation:
    rgb: NDArray[np.uint8]
    observation: NDArray[np.float32]
    action: int
    environment_reward: float


@dataclass()
class PairedObservations:
    obs_1: Slist[Observation]
    obs_2: Slist[Observation]
    preference: float | None = None
