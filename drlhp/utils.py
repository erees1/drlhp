import logging
import math
import random
from collections import defaultdict, deque

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger()


class ReplayBuffer:
    def __init__(self, maxlen, batch_size=32):
        self.max_len = maxlen
        self.buf = deque(maxlen=maxlen)
        self.batch_size = batch_size

    def sample_batch(self):
        # sample batch_size items from buffer
        batch = random.sample(self.buf, self.batch_size)

        # Each item to be its own tensor of len batch_size
        b = list(zip(*batch))
        batch = [np.array(t) for t in b]
        batch_as_tensors = [torch.from_numpy(t) for t in batch]
        out = []
        for t in batch_as_tensors:
            if t.dtype == torch.float64:
                t = t.float()
            out.append(t)
        return out

    def append(self, x):
        self.buf.append(x)

    def __getitem__(self, idx):
        return self.buf[idx]

    def __len__(self):
        return len(self.buf)


class Metrics:
    def __init__(self, prefix, summary_writer, log_every_step=1, log_every_ep=1):
        pass
        self.ep_metrics = defaultdict(list)
        self.step_metrics = {}
        self.agg_methods = {}
        self.possible_agg_methods = ["mean", "sum"]
        self.prefix = prefix
        self.log_every_step = log_every_step
        self.log_every_ep = log_every_ep

        self.writer = summary_writer

    def add_episode_metric(self, name, value, agg_method="mean"):
        if agg_method not in self.possible_agg_methods:
            raise ValueError(f"Aggregation method {agg_method} not supported")
        self.agg_methods[name] = agg_method
        self.ep_metrics[name].append(value)

    def add_step_metric(self, name, value):
        self.step_metrics[name] = value

    def log_step(self, step):
        for k, v in self.step_metrics.items():
            tag = f"{self.prefix}/{k}"
            self.writer.add_scalar(tag, v, step)
        self.step_metrics = {}

    def aggregate_episode(self):
        self.agg = {}
        for k, v in self.ep_metrics.items():
            if self.agg_methods[k] == "mean":
                self.agg[k] = np.mean(v)
            elif self.agg_methods[k] == "sum":
                self.agg[k] = np.sum(v)
        self.ep_metrics = defaultdict(list)

    def log_episode(self, episode, step):
        msg = f"{self.prefix} " if self.prefix else ""
        msg += f"episode: {episode} "
        msg += f"step: {step} "
        msg += " ".join([f"{k}: {v:.5f}" for k, v in self.agg.items()])

        if episode % self.log_every_ep == 0:
            logger.info(msg)

            for k, v in self.agg.items():
                tag = f"{self.prefix}_ep/{k}"
                self.writer.add_scalar(tag, v, episode)
                step_tag = f"{self.prefix}/{k}"
                self.writer.add_scalar(step_tag, v, step)


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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_dict(d, sep="."):
    # return list of tuples of (key.key2, value)
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.extend([(k + sep + k2, v2) for k2, v2 in flatten_dict(v, sep)])
        else:
            out.append((k, v))
    return out
