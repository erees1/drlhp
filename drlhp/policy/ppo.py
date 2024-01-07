import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from multiprocessing import Event, Queue
from typing import Callable, Generator, Optional

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from numpy.typing import NDArray
from slist import Slist
from torch import Tensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from drlhp.comms import Observation
from drlhp.envs.util import get_vec_env
from drlhp.models import MLP, AtariPolicy
from drlhp.policy.config import PPOConfig
from drlhp.utils.logging import log_config_to_tb, log_metrics, setup_logger
from drlhp.utils.optimizer import WarmupCA
from drlhp.utils.utils import get_observation_processing_func, reseed, seed_everything  # type: ignore


@dataclass
class StepMetaInformation:
    policy_loss: float
    value_loss: float
    policy_entropy: float
    loss: float


@dataclass
class FlatPPOTrajectory:
    observations: Tensor
    next_observations: Tensor
    actions: Tensor
    rewards: Tensor
    log_probs: Tensor
    dones: Tensor
    advantages: Tensor
    returns: Tensor


class StepMetaInformationHistory:
    def __init__(self, max_len: int):
        self.history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max_len))

    def add(self, meta: StepMetaInformation):
        for k, v in asdict(meta).items():
            self.history[f"{k}_avg"].append(v)

    def dict(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for k, v in self.history.items():
            output[k] = float(np.mean(list(v)))
        return output


def get_batches_from_trajectories(
    flattened_trajectories: FlatPPOTrajectory, batch_size: int
) -> Generator[FlatPPOTrajectory, None, None]:
    indices = np.random.permutation(len(flattened_trajectories.observations))
    for start_idx in range(0, len(flattened_trajectories.observations), batch_size):
        batch = {}
        end_idx = start_idx + batch_size
        minibatch_indices = indices[start_idx:end_idx]

        for k, v in asdict(flattened_trajectories).items():
            batch[k] = torch.from_numpy(np.array([v[i] for i in minibatch_indices]))

        yield FlatPPOTrajectory(**batch)


class PPOAgent:
    def __init__(self, observation_size: int, action_space: int, config: PPOConfig, env_name: str):
        self.config = config
        self.env_name = env_name
        self.action_space_size = action_space
        self.observation_space_size = observation_size
        self.gamma = config.gamma
        self.tau = config.tau
        if torch.backends.mps.is_available():  # type: ignore
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "CartPole" in env_name:
            self.critic_layer_sizes = [observation_size] + config.critic_n_layers * [config.critic_hidden_dim] + [1]
            self.actor_layer_sizes = (
                [observation_size] + config.actor_n_layers * [config.actor_hidden_dim] + [action_space]
            )
            self.critic = MLP(self.critic_layer_sizes).to(self.device)
            self.actor = MLP(self.actor_layer_sizes).to(self.device)
        else:
            self.critic = AtariPolicy(1).to(self.device)
            self.actor = AtariPolicy(action_space).to(self.device)

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
        scheduler_steps = int(config.n_timesteps * config.optimization_epochs / config.batch_size)
        self.scheduler = WarmupCA(
            self.optimizer, scheduler_steps, eta_min=1e-7, warmup_steps=config.lr_warmup_steps, decay_factor=10
        )

        self.steps_trained = 0

    def get_lr(self) -> list[float]:
        return self.scheduler.get_lr()

    def set_step_for_lr(self, step: int):
        self.scheduler.set_step(step)

    def compute_gae(self, observations: Tensor, next_observations: Tensor, rewards: Tensor, dones: Tensor):
        # atarti:
        # observations: (trajectory_length, n_envs, 4, 84, 84)
        # cartpole:
        # observations: (trajectory_length, n_envs, 4)
        traj_length, n_envs = observations.shape[:2]
        if len(observations.shape) == 5:
            observations = observations.view(-1, *observations.shape[2:])
            next_observations = next_observations.view(-1, *next_observations.shape[2:])

        observations = observations.to(self.device)
        next_observations = next_observations.to(self.device)

        values = self.critic(observations).cpu().detach().squeeze(-1)  # (trajectory_length, n_envs)
        next_values = self.critic(next_observations).cpu().detach().squeeze(-1)  # (trajectory_length, n_envs)

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

    def act(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        dist = self.get_action_dist(observation)
        action = dist.sample()
        return action.cpu(), dist.log_prob(action).cpu()

    def get_action_dist(self, observation: Tensor):
        observation = observation.to(self.device)
        logits_a = self.actor(observation)
        dist = torch.distributions.Categorical(logits=logits_a)
        return dist

    def update_step(
        self, observations: Tensor, actions: Tensor, returns: Tensor, advantages: Tensor, log_probs: Tensor
    ) -> StepMetaInformation:
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
        policy_entropy = -(new_log_probs.exp() * new_log_probs).mean()

        # Combine the losses and perform a gradient update
        loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * policy_entropy
        self.optimizer.zero_grad()
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)  # type: ignore
        self.optimizer.step()
        self.scheduler.step()

        self.steps_trained += 1

        meta = StepMetaInformation(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            policy_entropy=policy_entropy.item(),
            loss=loss.item(),
        )
        return meta

    def evaluate(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        assert len(actions.shape) == 1
        observations = observations.to(self.device)
        actions = actions.to(self.device)

        # Get the normalized log probabilities for the observations
        policy_log_probs = self.get_action_dist(observations)
        policy_log_probs = policy_log_probs.log_prob(actions)

        # Get the value estimates for the observations
        value_estimates = self.critic(observations).squeeze(-1)
        assert value_estimates.shape == policy_log_probs.shape

        return policy_log_probs.cpu(), value_estimates.cpu()


@dataclass
class Trajectories:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor


class Runner:
    """
    Iterable that collects trajectories from the environment and computes advantages and returns
    and returns flattened trajectories of torch tensors
    """

    def __init__(
        self,
        env: VectorEnv[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
        agent: PPOAgent,
        trajectory_length_per_env: int,
        process_func: Callable[[Tensor], Tensor],
        reward_func: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        self.env = env
        self.agent = agent
        self.trajectory_length = trajectory_length_per_env * env.num_envs
        self.single_env_length = trajectory_length_per_env
        self.renderings: Slist[Slist[Observation]] = Slist()
        self.current_rendering: Slist[Observation] = Slist()
        self.reward_func = reward_func

        self.process_func = process_func

    def collect_renderings(self) -> Slist[Slist[Observation]]:
        """
        Returns mutliple trajectories of observations for preferences as a list of lists
        """
        output = self.renderings
        self.renderings = Slist()
        return output

    def _collect_trajectories(self) -> Generator[tuple[Trajectories, list[int]], None, None]:
        observations, next_observations, actions, rewards, log_probs, dones = [], [], [], [], [], []
        ep_returns = []

        observation: NDArray[np.float32]  # shape: (n_envs, 4, 84, 84) for atari
        observation, _ = self.env.reset(seed=reseed())
        ep_reward = np.array([0.0] * self.env.num_envs)
        collect_rendering = True
        while True:
            action, log_prob = self.agent.act(self.process_func(torch.from_numpy(observation)))
            next_observation, env_reward, done, _, _ = self.env.step(action.cpu().numpy())
            if self.reward_func is not None:
                reward = self.reward_func(torch.from_numpy(next_observation)).numpy()
            else:
                reward = env_reward

            if collect_rendering:
                # shape (300, 300, 3) for atari, only the first env in envs is setup to render rgb for human inspection
                rgb: NDArray[np.uint8] = self.env.envs[0].render()  # type: ignore
                # Take the first observation, because this is a vectorized env so we are running multiple envs
                # and we just collect a trajectory from the first one
                obs_for_pref = Observation(rgb, observation[0], action.cpu().numpy()[0], env_reward[0])
                self.current_rendering.append(obs_for_pref)
                if done[0]:  # If the first one is done, we start a new trajectory
                    print("First env done")
                    print("starting a new collection")
                    self.renderings.append(self.current_rendering)
                    self.current_rendering = Slist()

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
                if done[0]:
                    # we collect the first full episode of the first environment
                    collect_rendering = True * random.random() < 1
                    print("Will collect next rendering", collect_rendering)

                for i in finished:
                    ep_returns.append(int(ep_reward[i]))
                    ep_reward[i] = 0

            if len(observations) >= self.single_env_length:
                trajectories = Trajectories(
                    torch.from_numpy(np.stack(observations, axis=0)),
                    torch.from_numpy(np.stack(next_observations, axis=0)),
                    torch.from_numpy(np.stack(actions, axis=0)),
                    torch.from_numpy(np.stack(rewards, axis=0)),
                    torch.from_numpy(np.stack(log_probs, axis=0)),
                    torch.from_numpy(np.stack(dones, axis=0)),
                )

                for v in asdict(trajectories).values():
                    assert v.shape[:2] == (self.single_env_length, self.env.num_envs)

                yield trajectories, ep_returns
                observations, next_observations, actions, rewards, log_probs, dones = [], [], [], [], [], []
                ep_returns = []

    def __iter__(self):
        for trajectories, ep_returns in self._collect_trajectories():
            # Compute advantages and returns
            advantages, returns = self.agent.compute_gae(
                self.process_func(trajectories.observations),
                self.process_func(trajectories.next_observations),
                trajectories.rewards,
                trajectories.dones,
            )

            def squeeze_tensor(v: Tensor) -> Tensor:
                remaining_dims = v.shape[2:]
                return v.view(v.shape[0] * v.shape[1], *remaining_dims)

            advantages = squeeze_tensor(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ft = FlatPPOTrajectory(
                observations=squeeze_tensor(self.process_func(trajectories.observations)),
                next_observations=squeeze_tensor(self.process_func(trajectories.next_observations)),
                actions=squeeze_tensor(trajectories.actions),
                rewards=squeeze_tensor(trajectories.rewards),
                log_probs=squeeze_tensor(trajectories.log_probs),
                dones=squeeze_tensor(trajectories.dones),
                advantages=advantages,
                returns=squeeze_tensor(returns),
            )

            yield ft, torch.tensor(ep_returns, dtype=torch.float32)


def train_ppo(
    env_name: str,
    config: PPOConfig,
    tb_dir: str,
    reward_func: Optional[Callable[[Tensor], Tensor]] = None,
    segment_queue: Optional[Queue] = None,  # type: ignore
    should_exit: Optional[Event] = None,  # type: ignore
):
    seed_everything(config.seed)
    writer = SummaryWriter(tb_dir)
    logger = setup_logger("ppo", level=logging.INFO)
    log_config_to_tb(config, writer)

    step_history = StepMetaInformationHistory(max_len=config.log_every)

    process_func = get_observation_processing_func(env_name)

    env = get_vec_env(env_name, n_envs=config.n_envs, render=True, use_async=config.use_async_env)
    observation_size: int = sum(env.single_observation_space.shape)  # type: ignore
    action_space: int = env.single_action_space.n  # type: ignore
    agent = PPOAgent(observation_size, action_space, config, env_name)
    runner = Runner(env, agent, config.steps_per_update_per_env, process_func, reward_func=reward_func)

    iteration = 0
    timesteps = 0

    start_collect = time.perf_counter()
    trajectories: FlatPPOTrajectory
    for trajectories, ep_returns in iter(runner):
        end_collect = time.perf_counter()
        timesteps += trajectories.observations.shape[0]
        renderings = runner.collect_renderings()
        if segment_queue is not None and len(renderings) > 0:
            print(f"put {len(renderings)} renderings on queue")
            renderings.for_each(lambda x: segment_queue.put(x))

        # Update policy and value function for multiple epochs using minibatches
        for _ in range(config.optimization_epochs):
            batch: FlatPPOTrajectory
            for batch in get_batches_from_trajectories(trajectories, config.batch_size):
                step_meta_information = agent.update_step(
                    batch.observations,
                    batch.actions,
                    batch.returns,
                    batch.advantages,
                    batch.log_probs,
                )
                if agent.steps_trained % config.log_every == 0:
                    step_history.add(step_meta_information)

                    step_meta_dict = asdict(step_meta_information)
                    step_meta_dict["env_steps"] = timesteps
                    step_meta_dict["pi_lr"], step_meta_dict["v_lr"] = agent.get_lr()

                    log_metrics(logger, step_meta_dict, agent.steps_trained, "updates", writer)
                    log_metrics(logger, step_history.dict(), agent.steps_trained, "updates", writer)

        end_train = time.perf_counter()
        trajectory_meta = {"avg_return": ep_returns.mean().item(), "n_episodes": len(ep_returns)}
        trajectory_meta["trajectory_time"] = end_collect - start_collect
        trajectory_meta["train_time"] = end_train - end_collect
        trajectory_meta["total_time"] = end_train - start_collect
        trajectory_meta["update_steps"] = agent.steps_trained
        trajectory_meta["env_steps"] = timesteps
        log_metrics(logger, trajectory_meta, iteration, "iter", writer=writer, log=True)

        iteration += 1
        if timesteps > config.n_timesteps:
            break

        start_collect = time.perf_counter()

    env.close()

    if should_exit is not None:
        should_exit.set()
    print("finished training")
    return
