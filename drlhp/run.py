import logging
import os
import random
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import yaml as yaml
from agents import get_agent_class
from fire import Fire
from torch.utils.tensorboard import SummaryWriter
from utils import Metrics, ReplayBuffer, seed_everything

logger = logging.getLogger()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


def environment_step(agent, env, observation, buffer_to_fill=None):
    action = agent.act(observation)
    next_observation, reward, done, truncation, _ = env.step(action)
    if buffer_to_fill is not None:
        buffer_to_fill.append((observation, action, reward, next_observation, done))
    return next_observation, done or truncation, reward


def parse_config(config):
    with open(config, "r") as f:
        return yaml.safe_load(f)


def log_config_to_tb(config, writer):
    def flatten_dict(d, sep="."):
        # return list of tuples of (key.key2, value)
        out = []
        for k, v in d.items():
            if isinstance(v, dict):
                out.extend([(k + sep + k2, v2) for k2, v2 in flatten_dict(v, sep)])
            else:
                out.append((k, v))
        return out

    for k, v in flatten_dict(config):
        try:
            v_as_num = float(v)
            writer.add_scalar(f"z_meta/{k}", v_as_num, 0)
        except ValueError:
            writer.add_text(f"z_meta/{k}", v)


def reseed():
    return random.randint(0, 1000000)


def override_config_with_kwargs(config, kwargs):
    for name, value in kwargs.items():
        # go down to right level
        d = config
        keys = name.split(".")
        for key in keys[:-1]:
            try:
                d = d[key]
            except KeyError as e:
                raise KeyError(f"Recieved input key {name} but {key} not found in config") from e
        # assign value, using the fact that dictionaries are mutable
        final_key = keys[-1]
        if final_key not in d:
            raise KeyError(f"Recieved input key {name} but {final_key} not found in config")
        d[keys[-1]] = value

    return config


def validate_on_hold_out_states(agent, val_bsz, hold_out_states, validation_metrics):
    all_Q_values = []
    for state in hold_out_states.split(val_bsz):
        Q_values = agent.get_max_Q_values(state)
        all_Q_values.append(Q_values)
    all_Q_values = torch.cat(all_Q_values)
    validation_metrics.add_step_metric("holdout_mean_Q", all_Q_values.mean().item())


def train(
    env: str = "CartPole-v1",
    config: str = "./drlhp/configs/dqn_fixed_target.yaml",
    exp_root="./exp",
    **kwargs,
):

    config = parse_config(config)
    config = override_config_with_kwargs(config, kwargs)

    current_date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = f"{exp_root}/{env}/{current_date_and_time}"
    key_params = []
    key_params.append(f"algo-{config['agent']['type']}")
    key_params.append(f"loss-{config['agent']['loss']}")
    key_params.append(f"seed-{config['seed']}")
    exp_dir += f"_{'_'.join(key_params)}"
    tb_dir = f"{exp_dir}/tb"

    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(tb_dir)

    # Log config to exp dir and tensorboard
    with open(f"{exp_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    log_config_to_tb(config, tb_writer)

    training_config = config["training"]
    seed_everything(config["seed"])

    # Setup
    env = gym.make(env, render_mode=None)
    observation_size = sum(env.observation_space.shape)
    action_space = env.action_space.n
    Agent = get_agent_class(config["agent"]["type"])
    agent = Agent(observation_size, action_space, config["agent"])
    replay_buffer = ReplayBuffer(config["replay_buffer_size"], training_config["batch_size"])

    # Create hold out set of states for validation
    hold_out_states = []
    while len(hold_out_states) < training_config["n_val_states"]:
        observation, _ = env.reset(seed=reseed())
        done = False
        while not done:
            observation, done, _ = environment_step(agent, env, observation)
            hold_out_states.append(observation)
    hold_out_states = np.array(hold_out_states)[: training_config["n_val_states"]]
    hold_out_states = torch.from_numpy(hold_out_states).to(torch.float32)

    # Warmup steps to fill replay buffer
    while len(replay_buffer) < replay_buffer.max_len:
        observation, _ = env.reset(seed=reseed())
        done = False
        while not done:
            observation, done, _ = environment_step(agent, env, observation, replay_buffer)

    training_metrics = Metrics("train", tb_writer)
    validation_metrics = Metrics("val", tb_writer)

    # Training loop
    i_episode = 0
    while True:
        observation, _ = env.reset(seed=reseed())

        done = False
        while not done:
            observation, done, reward = environment_step(agent, env, observation, replay_buffer)
            agent.exploration_schedule.step()

            batch = replay_buffer.sample_batch()
            agent.update(batch, training_metrics)
            training_metrics.add_episode_metric("episode_reward", reward, "sum")

            if agent.steps_trained % training_config["log_step_every"] == 0:
                training_metrics.log_step(agent.steps_trained)

            if agent.steps_trained % training_config["val_every"] == 0:
                validate_on_hold_out_states(
                    agent,
                    training_config["batch_size"],
                    hold_out_states,
                    validation_metrics,
                )
                validation_metrics.log_step(agent.steps_trained)

        training_metrics.aggregate_episode()
        if i_episode % training_config["log_ep_every"] == 0:
            training_metrics.log_episode(i_episode, agent.steps_trained)
        i_episode += 1

        if agent.steps_trained > training_config["n_steps"]:
            break

    env.close()


if __name__ == "__main__":
    Fire(train)
