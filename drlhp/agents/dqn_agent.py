import math
import random

import torch
import torch.nn.functional as F
from models import MLP
from utils import Metrics, WarmupCA


class EpsilonSchedule:
    def __init__(self, profile, epsilon_init: int, decay_steps: int, epsilon_min: int):
        self.profile = profile
        self.epsilon_init = epsilon_init
        self.decay_steps = decay_steps
        self.epsilon_min = epsilon_min
        self._steps = 0

    def step(self):
        self._steps += 1

    @property
    def epsilon(self):
        if self.profile == "linear":
            return self.linear()
        elif self.profile == "exponential":
            return self.exponential()
        else:
            raise ValueError(f"Unknown epsilon profile {self.profile}")

    def linear(self):
        epsilon = self.epsilon_init - (self.epsilon_init - self.epsilon_min) * self._steps / self.decay_steps
        return max(epsilon, self.epsilon_min)

    def exponential(self):
        return self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(
            -1.0 * self._steps / self.decay_steps
        )


class DQNAgent:
    def __init__(self, observation_size: int, action_space: int, config: dict):

        self.action_space_size = action_space
        self.observation_space_size = observation_size
        self.layer_sizes = (
            [observation_size] + config["model"]["n_layers"] * [config["model"]["hidden_dim"]] + [action_space]
        )
        self.m = MLP(self.layer_sizes)

        optimizer_params = config["optimizer"]
        init_lr = float(optimizer_params["lr"])
        if optimizer_params["name"] == "adam":
            self.optimizer = torch.optim.Adam(self.m.parameters(), lr=init_lr)
        elif optimizer_params["name"] == "AdamW":
            self.optimizer = torch.optim.AdamW(self.m.parameters(), lr=init_lr)
        else:
            raise ValueError(f"Unknown optimizer {optimizer_params['name']}")

        optimizer_decay_steps = optimizer_params["decay_steps"]
        warmup_steps = optimizer_params["warmup_steps"]
        self.learning_rate_scheduler = WarmupCA(self.optimizer, optimizer_decay_steps, warmup_steps=warmup_steps)

        self.exploration_schedule = EpsilonSchedule(**config["exploration"])
        self.discount_factor = config["discount_factor"]
        self.max_grad_norm = config["max_grad_norm"]

        if config["loss"] == "mse":
            self.loss_f = F.mse_loss
        elif config["loss"] == "huber":
            self.loss_f = torch.nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss {config['loss']}")

        self.steps_trained = 0
        self.set_explore(True)

    def set_explore(self, explore: bool):
        self.explore = explore

    def training(self):
        self.m.train()

    def eval(self):
        self.m.eval()

    def act(self, observation):

        epsilon = self.exploration_schedule.epsilon

        if self.explore and random.random() < epsilon:
            # Sample random action
            action = random.randint(0, self.action_space_size - 1)
        else:
            Q = self.m(torch.from_numpy(observation))
            action = torch.argmax(Q)

        return int(action)

    def update(self, batch, metrics: Metrics):

        obs, actions, r, next_obs, dones = batch

        pred_Q_values = self.m(obs).gather(1, actions.unsqueeze(-1)).squeeze()
        target_Q_values = self.get_target_Q_values(next_obs, dones, r)

        torch.nn.utils.clip_grad_norm_(self.m.parameters(), self.max_grad_norm)

        loss = self.loss_f(pred_Q_values, target_Q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        # record relevant metrics for logging and tensorboard
        metrics.add_episode_metric("episode_mean_Q", torch.mean(pred_Q_values).item(), "mean")
        metrics.add_episode_metric("episode_mean_loss", loss.item(), "mean")
        metrics.add_step_metric("loss", loss.item())
        metrics.add_step_metric("epsilon", self.exploration_schedule.epsilon)
        metrics.add_step_metric("lr", self.optimizer.param_groups[0]["lr"])

        self.steps_trained += 1

    def get_target_Q_values(self, obs, dones, r):
        max_next_Q_values = self.m(obs).detach().max(axis=-1).values
        target_Q_values = r + (1 - dones * 1) * self.discount_factor * max_next_Q_values
        return target_Q_values

    def get_max_Q_values(self, obs):
        Q_values = self.m(obs)
        max_Q_values = torch.max(Q_values, axis=-1).values
        return max_Q_values

    def save_model(self, path):
        torch.save(self.m.state_dict(), path)

    def load_model(self, path):
        self.m.load_state_dict(torch.load(path))


class DQNFixedTarget(DQNAgent):
    def __init__(self, observation_size: int, action_space: int, config: dict):
        super().__init__(observation_size, action_space, config)
        self.target_m = MLP(self.layer_sizes)
        self.target_m.load_state_dict(self.m.state_dict())
        self.target_m.eval()
        self.tau = config["tau"]

    def get_target_Q_values(self, obs, dones, r):
        max_next_Q_values = self.target_m(obs).detach().max(axis=-1).values
        target_Q_values = r + (1 - dones * 1) * self.discount_factor * max_next_Q_values
        self.update_target()
        return target_Q_values

    def update_target(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_m.state_dict()
        policy_net_state_dict = self.m.state_dict()
        for key in policy_net_state_dict:  # pylint: disable=not-an-iterable
            new_weights = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            target_net_state_dict[key] = new_weights  # pylint: disable=unsupported-assignment-operation
        self.target_m.load_state_dict(target_net_state_dict)
