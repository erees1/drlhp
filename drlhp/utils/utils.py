import random
from functools import partial
from typing import Callable

import numpy as np
import torch


def reseed():
    return random.randint(0, 1000000)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _process_observations(obs: torch.Tensor, add_batch_dim: bool = False, scale: bool = False) -> torch.Tensor:
    if scale:
        obs = obs.float() / 255.0

    if add_batch_dim:
        obs = obs.unsqueeze(0)
    return obs


def get_observation_processing_func(env_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if "Cart" in env_name:
        return _process_observations
    else:
        return partial(_process_observations, scale=True)
