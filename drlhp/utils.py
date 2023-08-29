from dataclasses import asdict, dataclass
import logging
import math
import random
from functools import partial
from typing import Callable, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from drlhp.config import PPOConfig  # type: ignore

logger = logging.getLogger()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
logger = logging.getLogger()


def reseed():
    return random.randint(0, 1000000)


class WarmupCA(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        steps: int,
        eta_min=1e-7,
        warmup_steps=2000,
        decay_factor=10,
    ):
        self.steps = steps
        self.eta_min = eta_min
        self.start_step = warmup_steps
        self.T_max = self.steps - self.start_step
        self.decay_factor = decay_factor

        super().__init__(optimizer, -1)

    def main_lr(self, base_lr: float, step: int) -> float:
        return (base_lr / self.decay_factor) + (base_lr - (base_lr / self.decay_factor)) * (
            1 + math.cos(math.pi * (step - self.start_step) / self.T_max)
        ) / 2

    def get_lr(self) -> list[float]:  # type: ignore
        lr_list = []

        for base_lr in self.base_lrs:
            # linear at first
            if 0 <= self._step_count < self.start_step:
                m = (base_lr - self.eta_min) / self.start_step
                lr_list.append(self.eta_min + m * self._step_count)
            # cosine decay
            else:
                lr_list.append(self.main_lr(base_lr, self._step_count))
        return lr_list

    def set_step(self, step: int):
        self._step_count = step


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_metrics(
    meta: dict[str, float | int], step: int, prefix: str, writer: Optional[SummaryWriter] = None, log: bool = False
):
    if log:
        msg = f"{prefix}: {step} "
        for k, v in meta.items():
            if isinstance(v, float):
                msg += f"{k}: {v:.4f} "
            else:
                msg += f"{k}: {v} "
        logger.info(msg)

    for k, v in meta.items():
        tag = f"train-{prefix}/{k}"
        if math.isnan(v):
            # handles the case where avg_reward is nan
            # because episode didn't finish
            continue
        if writer is not None:
            writer.add_scalar(tag, v, step)


def _get_indvidual_env(
    env_name: str,
    render: bool = False,
) -> gym.Env | gym.wrappers.FrameStack:  # type: ignore
    if render:
        render_mode = "rgb_array"
    else:
        render_mode = None

    if "Pong" in env_name:
        kwargs = dict(grayscale_obs=True)

        env = gym.make(f"{env_name}", render_mode=render_mode)
        # Atari preprocessing wrapper
        env = gym.wrappers.AtariPreprocessing(  # type: ignore
            env,
            noop_max=0,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_newaxis=False,
            scale_obs=False,
            **kwargs,
        )
        # Frame stacking
        env: gym.wrappers.FrameStack = gym.wrappers.FrameStack(env, 4)  # type: ignore
        return env

    elif "MovingDot" in env_name:
        env: gym.Env = gym.make(  # type: ignore
            env_name,
            render_mode=render_mode,
            size=(84, 84),
            channel_dim=False,
            max_steps=300,
        )
        env: gym.wrappers.FrameStack = gym.wrappers.FrameStack(env, 4)  # type: ignore

    else:
        env: gym.Env = gym.make(env_name, render_mode=render_mode)  # type: ignore
    return env


def get_vec_env(env_name: str, render: bool = False, n_envs: int = 1, use_async: bool = False) -> gym.vector.VectorEnv:
    envs = [partial(_get_indvidual_env, env_name, render=bool((i == 0) * render)) for i in range(n_envs)]
    if use_async:
        envs = gym.vector.AsyncVectorEnv(envs)
    else:
        envs = gym.vector.SyncVectorEnv(envs)

    envs.env_name = env_name  # type: ignore
    return envs


def get_env_name(env: Union[gym.vector.AsyncVectorEnv, gym.vector.SyncVectorEnv]) -> str:
    if isinstance(env, gym.vector.AsyncVectorEnv) or isinstance(env, gym.vector.SyncVectorEnv):
        return env.envs[0].spec.id  # type: ignore
    else:
        return env.spec.id


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


@dataclass
class Trajectories:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor


def log_config_to_tb(config: PPOConfig, writer: SummaryWriter):
    for k, v in asdict(config).items():
        try:
            v_as_num = float(v)
            writer.add_scalar(f"z_meta/{k}", v_as_num, 0)
        except ValueError:
            writer.add_text(f"z_meta/{k}", v)
