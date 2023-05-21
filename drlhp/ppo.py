import time
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from models import CNN, MLP
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import (
    WarmupCA,
    get_batches_from_trajectories,
    get_env_name,
    get_vec_env,
    log_metrics,
    process_observations,
    reseed,
)


@dataclass
class PPOConfig:
    optimizer: str = "AdamW"
    pi_lr: float = 1e-5
    v_lr: float = 1e-4
    lr_warmup_steps: int = 1000
    actor_hidden_dim: int = 128
    actor_n_layers: int = 1
    critic_hidden_dim: int = 128
    critic_n_layers: int = 1
    seed: int = 1
    steps_per_update: int = 6400
    n_timesteps: int = 1e6
    optimization_epochs: int = 10
    batch_size: int = 32
    log_every: int = 100
    entropy_coeff: float = 0.01
    clip_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    render_every: int = 5
    n_envs: int = 32
    gamma: float = 0.99
    tau: float = 0.95
    grad_clip: float = 0.5


class PPOAgent:
    def __init__(self, observation_size: int, action_space: int, config: PPOConfig, env_name: str):
        self.config = config
        self.env_name = env_name
        self.action_space_size = action_space
        self.observation_space_size = observation_size
        self.gamma = config.gamma
        self.tau = config.tau

        if "CartPole" in env_name:
            self.critic_layer_sizes = [observation_size] + config.critic_n_layers * [config.critic_hidden_dim] + [1]
            self.actor_layer_sizes = (
                [observation_size] + config.actor_n_layers * [config.actor_hidden_dim] + [action_space]
            )
            self.critic = MLP(self.critic_layer_sizes)
            self.actor = MLP(self.actor_layer_sizes)
        else:
            self.critic = CNN(1)
            self.actor = CNN(action_space)

        self.entropy_coeff = config.entropy_coeff
        self.clip_epsilon = config.clip_epsilon
        self.value_loss_coeff = config.value_loss_coeff

        params = [
            {"params": self.actor.parameters(), "lr": config.pi_lr},
            {"params": self.critic.parameters(), "lr": config.v_lr},
        ]

        if config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(params)
        elif config.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(params)
        else:
            raise ValueError(f"Unknown optimizer {config.optimizer}")

        # scheudler is stepped every update so the max num of updates is equal to
        #  config.n_timesteps * config.optimization_epochs / bsz
        scheduler_steps = config.n_timesteps * config.optimization_epochs / config.batch_size
        self.scheduler = WarmupCA(self.optimizer, scheduler_steps, eta_min=1e-7, warmup_steps=2000, decay_factor=10)

        self.steps_trained = 0

    def get_lr(self):
        return self.scheduler.get_lr()

    def set_step_for_lr(self, step):
        self.scheduler.set_step(step)

    def compute_gae(self, observations, next_observations, rewards, dones):
        # atarti:
        # observations: (trajectory_length, n_envs, 4, 84, 84)
        # cartpole:
        # observations: (trajectory_length, n_envs, 4)
        if len(observations.shape) == 5:
            traj_length, n_envs = observations.shape[:2]
            observations = observations.view(-1, *observations.shape[2:])
            next_observations = next_observations.view(-1, *next_observations.shape[2:])

        values = self.critic(observations).detach().squeeze(-1)  # (trajectory_length, n_envs)
        next_values = self.critic(next_observations).detach().squeeze(-1)  # (trajectory_length, n_envs)

        if len(values.shape) == 1:
            values = values.view(traj_length, n_envs)
            next_values = next_values.view(traj_length, n_envs)

        assert rewards.shape == values.shape
        assert next_values.shape == values.shape

        # Calculate TD errors
        # (traj_len, n_envs)
        td_errors = rewards + self.gamma * next_values * (1 - 1 * dones) - values

        # Calculate GAE
        advantages = torch.zeros_like(td_errors)
        gae = 0
        for t in reversed(range(len(td_errors))):
            gae = td_errors[t] + self.gamma * self.tau * (1 - 1 * dones[t]) * gae
            advantages[t] = gae

        # Calculate returns
        returns = advantages + values

        return advantages, returns

    def act(self, observation: torch.Tensor):
        dist = self.get_action_dist(observation)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_action_dist(self, observation: torch.Tensor):
        logits_a = self.actor(observation)
        dist = torch.distributions.Categorical(logits=logits_a)
        return dist

    def update_step(self, observations, actions, returns, advantages, log_probs):
        new_log_probs, values = self.evaluate(observations, actions)

        # Calculate the policy ratio
        ratio = (new_log_probs - log_probs).exp()

        # Calculate the surrogate loss
        surrogate_loss_1 = ratio * advantages
        surrogate_loss_2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate_loss_1, surrogate_loss_2).mean()

        # Calculate the value loss using a Huber loss (smooth L1 loss)
        value_loss = F.smooth_l1_loss(values, returns)

        # Calculate the entropy bonus
        entropy_bonus = -(new_log_probs.exp() * new_log_probs).mean()

        # Combine the losses and perform a gradient update
        loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy_bonus
        self.optimizer.zero_grad()
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        self.steps_trained += 1

        # Update training metrics
        meta = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_bonus": entropy_bonus.item(),
            "loss": loss.item(),
        }
        return meta

    def evaluate(self, observations, actions):
        assert len(actions.shape) == 1
        # Get the normalized log probabilities for the observations
        policy_log_probs = self.get_action_dist(observations)
        policy_log_probs = policy_log_probs.log_prob(actions)

        # Get the value estimates for the observations
        value_estimates = self.critic(observations).squeeze(-1)
        assert value_estimates.shape == policy_log_probs.shape

        return policy_log_probs, value_estimates


class Runner:
    def __init__(self, env, agent: PPOAgent, trajectory_length):
        self.env = env
        self.agent = agent
        self.trajectory_length = trajectory_length
        self.single_env_length = trajectory_length // env.num_envs

        env_name = get_env_name(env)
        if "Pong" in env_name:
            self.process_func = partial(process_observations, scale=True)
        else:
            self.process_func = process_observations

    def _flatten_trajectory(self, trajectories):
        # squeeze the trajectory and n_envs dimension into 1
        flattened_trajectories = {}
        for k, v in trajectories.items():
            remaining_dims = v.shape[2:]
            flattened_trajectories[k] = v.view(v.shape[0] * v.shape[1], *remaining_dims)
        return flattened_trajectories

    def _collect_trajectories(self):
        observations, next_observations, actions, rewards, log_probs, dones = [], [], [], [], [], []
        ep_returns = []

        observation, _ = self.env.reset(seed=reseed())
        ep_reward = np.array([0.0] * self.env.num_envs)
        while True:

            action, log_prob = self.agent.act(self.process_func(observation))
            next_observation, reward, done, truncation, _ = self.env.step(action.numpy())

            # Store trajectory data
            observations.append(observation)
            next_observations.append(next_observation)
            actions.append(action)
            ep_reward += reward
            rewards.append(reward)
            log_probs.append(log_prob.detach().numpy())
            dones.append(done)

            observation = next_observation
            # if any env finished
            if done.any():
                # finished envs
                finished = np.argwhere(done)
                for i in finished:
                    ep_returns.append(int(ep_reward[i]))
                    ep_reward[i] = 0

            if len(observations) >= self.single_env_length:
                trajectories = {
                    "observations": observations,
                    "next_observations": next_observations,
                    "actions": actions,
                    "rewards": rewards,
                    "log_probs": log_probs,
                    "dones": dones,
                }
                yield trajectories, ep_returns
                observations, next_observations, actions, rewards, log_probs, dones = [], [], [], [], [], []
                ep_returns = []

    def __iter__(self):
        for trajectories, ep_returns in self._collect_trajectories():
            for k, v in trajectories.items():
                arr = np.stack(v, axis=0)  # (trajectory_length, n_envs, ...)
                assert arr.shape[:2] == (self.single_env_length, self.env.num_envs)
                trajectories[k] = torch.from_numpy(arr)

            advantages, returns = self.agent.compute_gae(
                self.process_func(trajectories["observations"]),
                self.process_func(trajectories["next_observations"]),
                trajectories["rewards"],
                trajectories["dones"],
            )
            trajectories["advantages"] = advantages
            trajectories["returns"] = returns

            flattened_trajectories = self._flatten_trajectory(trajectories)
            advantages = flattened_trajectories["advantages"]
            flattened_trajectories["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            yield flattened_trajectories, torch.tensor(ep_returns, dtype=torch.float32)


class MetaHistory:
    def __init__(self, max_len):
        self.meta = defaultdict(lambda: deque(maxlen=max_len))

    def add(self, meta):
        for k, v in meta.items():
            self.meta[f"{k}_avg"].append(v)

    def items(self):
        output = {}
        for k, v in self.meta.items():
            output[k] = np.mean(v)
        return output.items()


def train_ppo(env_name, config: PPOConfig, writer: SummaryWriter):

    meta_history = MetaHistory(max_len=config.log_every)

    env = get_vec_env(env_name, n_envs=config.n_envs)
    observation_size = sum(env.single_observation_space.shape)
    action_space = env.single_action_space.n
    agent = PPOAgent(observation_size, action_space, config, env_name)
    runner = Runner(env, agent, config.steps_per_update)

    iteration = 0
    timesteps = 0

    start_collect = time.perf_counter()
    for trajectories, ep_returns in iter(runner):
        end_collect = time.perf_counter()
        timesteps += trajectories["observations"].shape[0]

        # Update policy and value function for multiple epochs using minibatches
        for _ in range(config.optimization_epochs):
            for batch in get_batches_from_trajectories(trajectories, config.batch_size):
                meta = agent.update_step(
                    batch["observations"],
                    batch["actions"],
                    batch["returns"],
                    batch["advantages"],
                    batch["log_probs"],
                )
                if agent.steps_trained % config.log_every == 0:
                    meta_history.add(meta)
                    meta["pi_lr"], meta["v_lr"] = agent.get_lr()
                    meta["env_steps"] = timesteps
                    log_metrics(meta, agent.steps_trained, "updates", writer)
                    log_metrics(meta_history, agent.steps_trained, "updates", writer)

        end_train = time.perf_counter()
        trajectory_meta = {"avg_return": ep_returns.mean().item(), "n_episodes": len(ep_returns)}
        trajectory_meta["trajectory_time"] = end_collect - start_collect
        trajectory_meta["train_time"] = end_train - end_collect
        trajectory_meta["total_time"] = end_train - start_collect
        trajectory_meta["update_steps"] = agent.steps_trained
        trajectory_meta["env_steps"] = timesteps
        log_metrics(trajectory_meta, iteration, "iter", writer=writer, log=True)

        iteration += 1
        if timesteps > config.n_timesteps:
            break

        start_collect = time.perf_counter()
