"""
Test all envs implemented over small number of steps
"""

import gymnasium as gym
import pytest

from drlhp.envs import gym_moving_dot  # noqa

ENVS = [
    "MovingDotDiscrete-v0",
    "MovingDotDiscreteNoFrameskip-v0",
    "MovingDotContinuous-v0",
    "MovingDotContinuousNoFrameskip-v0",
]


@pytest.mark.parametrize("env_name", ENVS)
def test_moving_dot_env(env_name: str, render: bool = False):
    print(f"=== Test: {env_name} ===")

    env = gym.make(env_name, render_mode="human")

    env.reset()

    for _ in range(3):
        a = env.action_space.sample()
        print(a)
        o, r, d, _, info = env.step(a)
        if render:
            env.render()
        print(f"Obs shape: {o.shape}, Action: {a}, Reward: {r}, Done flag: {d}, Info: {info}")

    env.close()
    del env


@pytest.mark.parametrize("env_name", ENVS)
def test_changing_seed_changes_observation(env_name: str):
    env = gym.make(env_name, render_mode="human")

    o1, _ = env.reset(seed=0)
    o2, _ = env.reset(seed=1)
    o3, _ = env.reset(seed=0)

    assert not (o1 == o2).all()
    assert (o1 == o3).all()

    env.close()
    del env


if __name__ == "__main__":
    for env in ENVS:
        test_moving_dot_env(env, render=True)
