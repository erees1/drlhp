import torch
from torch.optim.lr_scheduler import _LRScheduler


import math


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
