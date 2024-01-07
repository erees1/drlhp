import logging
import math
from typing import Optional
from drlhp.policy.config import PPOConfig


from torch.utils.tensorboard import SummaryWriter


from dataclasses import asdict


def log_config_to_tb(config: PPOConfig, writer: SummaryWriter):
    for k, v in asdict(config).items():
        try:
            v_as_num = float(v)
            writer.add_scalar(f"z_meta/{k}", v_as_num, 0)
        except ValueError:
            writer.add_text(f"z_meta/{k}", v)


def log_metrics(
    logger: logging.Logger,
    meta: dict[str, float | int],
    step: int,
    prefix: str,
    writer: Optional[SummaryWriter] = None,
    log: bool = False,
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


def setup_logger(name: str, level: int = logging.WARNING):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
