import itertools
from multiprocessing import Event, Process, Queue
import os
from dataclasses import asdict
from datetime import datetime
from time import sleep
from typing import Optional, Any
from drlhp.config import PPOConfig
from drlhp.web_interface.app import preference_interface_loop
from drlhp import gym_moving_dot  # noqa

import yaml as yaml
from fire import Fire
from drlhp.reward_predictor import return_ones, reward_interface_loop
from drlhp.ppo import train_ppo


def make_exp_dir_path(
    exp_root: str,
    env: str,
    algo: str,
    config: PPOConfig,
    exp_name: Optional[str] = None,
    passed_kwargs: Optional[dict[str, Any]] = None,
):
    if passed_kwargs is None:
        passed_kwargs = {}
    # Construt the exp dir path similar to openai spinup
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H-%M-%S")
    if exp_name is None:
        exp_name = f"{env}_{algo}"
    parent_dir = f"{current_date}_{exp_name}"
    inner_dir = f"{current_time}"

    for k, v in passed_kwargs.items():
        if k == "seed":
            continue
        inner_dir += f"_{k}{v}"

    exp_dir = f"{exp_root}/{parent_dir}/{inner_dir}/{config.seed}"
    return exp_dir


def worker_function(number):
    print(f"Worker {number} is working.")


def train(
    env: str,
    algo: str,
    exp_root: str,
    exp_name: Optional[str] = None,
    help: bool = False,
    use_reward_func: bool = False,
    **kwargs,  # type: ignore
):
    Config = PPOConfig

    if help:
        print(f"Available parameters for selected algo {algo}:")
        for field in Config.__annotations__.keys():
            print(f"{field}: default: {getattr(Config, field)}")
        exit(0)

    config = Config(**kwargs)

    # Make a sensible experiment directory name with passed kwargs and seed as suffix
    exp_dir = make_exp_dir_path(exp_root, env, algo, config, exp_name=exp_name, passed_kwargs=kwargs)
    tb_dir = f"{exp_dir}/tb"
    os.makedirs(tb_dir, exist_ok=True)

    # Log config to exp dir and tensorboard
    with open(f"{exp_dir}/config.yaml", "w") as f:
        yaml.dump(asdict(config), f)
    with open(f"{exp_dir}/env", "w") as f:
        print(env, file=f)

    if use_reward_func:
        reward_func = return_ones
        SegmentDatabase(size=1000)
    else:
        reward_func = None

    if False:
        train_ppo(
            env,
            config,
            tb_writer,
            reward_func=reward_func,
        )

    reward_func = None
    # set up our processes
    # 1. process for training / collecting trajectories
    # # 2. another that recieves them
    segment_queue = Queue(maxsize=100)
    preference_queue = Queue(maxsize=100)
    should_exit = Event()

    policy_process = Process(target=train_ppo, args=(env, config, tb_dir, reward_func, segment_queue, should_exit))
    preference_process = Process(target=preference_interface_loop, args=(segment_queue, preference_queue, should_exit))
    reward_process = Process(target=reward_interface_loop, args=(preference_queue, should_exit))

    reward_process.start()
    preference_process.start()
    sleep(3)
    policy_process.start()

    preference_process.join()
    policy_process.join()
    reward_process.join()


def create_param_grid(kwargs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:  # type: ignore
    # Get all parameters in kwargs that are tuples
    combinations = []
    sweep_parameters: list[str] = []
    for k, v in kwargs.items():
        if isinstance(v, tuple):
            if k not in sweep_parameters:
                sweep_parameters.append(k)
            values_to_try = []
            for v_ in v:
                values_to_try.append((k, v_))
            combinations.append(values_to_try)

    # For every possible combination of parameters, create a new set of kwargs
    param_sets = list(itertools.product(*combinations))
    new_kwargs = []
    for param_set in param_sets:
        new_kwargs.append(kwargs.copy())
        for k, v in param_set:
            new_kwargs[-1][k] = v

    return new_kwargs, sweep_parameters

    # env: str = "PongNoFrameskip-v4",


def main(
    algo: str = "ppo",
    env: str = "MovingDotDiscreteNoFrameskip-v0",
    exp_name: Optional[str] = None,
    exp_root: str = "./exp",
    config_help: bool = False,
    use_reward_func: bool = False,
    **kwargs,  # type: ignore
):
    param_sets, _ = create_param_grid(kwargs)
    for kwargs in param_sets:
        train(
            env=env,
            algo=algo,
            exp_name=exp_name,
            exp_root=exp_root,
            help=config_help,
            use_reward_func=use_reward_func,
            **kwargs,  # type: ignore
        )


if __name__ == "__main__":
    Fire(main)
