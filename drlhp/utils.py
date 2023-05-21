import logging
import math
import random
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

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
        optimizer,
        steps,
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

    def main_lr(self, base_lr, step):
        return (base_lr / self.decay_factor) + (base_lr - (base_lr / self.decay_factor)) * (
            1 + math.cos(math.pi * (step - self.start_step) / self.T_max)
        ) / 2

    def get_lr(self):
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

    def set_step(self, step):
        self._step_count = step


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_metrics(meta: dict, step: int, prefix, writer: SummaryWriter = None, log=False):
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
        writer.add_scalar(tag, v, step)


def _get_indvidual_env(env_name, render=False, n_envs=1):
    render_mode = "rgb_array" if render else None
    if "Pong" in env_name:
        env = gym.make(f"{env_name}", render_mode=render_mode)
        # Atari preprocessing wrapper
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )
        # Frame stacking
        env = gym.wrappers.FrameStack(env, 4)
        return env

    env = gym.make(env_name, render_mode=render_mode)
    return env


def get_vec_env(env_name, render=False, n_envs=1, use_async=False):
    envs = [partial(_get_indvidual_env, env_name, render=(i == 0) * render) for i in range(n_envs)]
    if use_async:
        envs = gym.vector.AsyncVectorEnv(envs)
    else:
        envs = gym.vector.SyncVectorEnv(envs)
    envs.env_name = env_name
    return envs


def get_env_name(env):
    if isinstance(env, gym.vector.AsyncVectorEnv) or isinstance(env, gym.vector.SyncVectorEnv):
        return env.envs[0].spec.id
    else:
        return env.spec.id


def get_batches_from_trajectories(flattened_trajectories, batch_size):

    indices = np.random.permutation(len(flattened_trajectories["observations"]))
    for start_idx in range(0, len(flattened_trajectories["observations"]), batch_size):
        batch = {}
        end_idx = start_idx + batch_size
        minibatch_indices = indices[start_idx:end_idx]

        for k, v in flattened_trajectories.items():
            if isinstance(v, list):
                batch[k] = torch.from_numpy(np.array([v[i] for i in minibatch_indices]))
            else:
                batch[k] = v[minibatch_indices]
        yield batch


def process_observations(obs, add_batch_dim=False, scale=False):

    # convert to torch tensor
    obs = torch.from_numpy(np.array(obs))

    if scale:
        obs = obs.float() / 255.0

    if add_batch_dim:
        obs = obs.unsqueeze(0)
    return obs
