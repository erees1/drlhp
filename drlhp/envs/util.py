from typing import Union
import gymnasium as gym

from functools import partial

from numpy.typing import NDArray
import numpy as np

from drlhp.envs import gym_moving_dot  # noqa


def _get_indvidual_env(
    env_name: str,
    render: bool = False,
) -> gym.Env[NDArray[np.float32], NDArray[np.float32]] | gym.wrappers.FrameStack:  # type: ignore
    if render:
        render_mode = "rgb_array"
    else:
        render_mode = None

    if "Pong" in env_name:
        kwargs = dict(grayscale_obs=True)

        env = gym.make(f"{env_name}", render_mode=render_mode)
        # Atari preprocessing wrapper
        env = gym.wrappers.AtariPreprocessing(  # type: ignore
            env,
            noop_max=0,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_newaxis=False,
            scale_obs=False,
            **kwargs,
        )
        # Frame stacking
        env: gym.wrappers.FrameStack = gym.wrappers.FrameStack(env, 4)  # type: ignore
        return env

    elif "MovingDot" in env_name:
        print("env_name", env_name)
        env: gym.Env = gym.make(  # type: ignore
            env_name,
            render_mode=render_mode,
            size=(84, 84),
            channel_dim=False,
            max_steps=300,
        )
        env: gym.wrappers.FrameStack = gym.wrappers.FrameStack(env, 4)  # type: ignore

    else:
        env: gym.Env = gym.make(env_name, render_mode=render_mode)  # type: ignore
    return env


def get_vec_env(
    env_name: str, render: bool = False, n_envs: int = 1, use_async: bool = False
) -> gym.vector.VectorEnv[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    if "Dot" in env_name:
        ienvs = [partial(_get_indvidual_env, env_name, render=bool((i == 0) * render)) for i in range(n_envs)]
    elif "Hopper" in env_name:
        ienvs = [partial(gym.make, env_name, render_mode="rgb_array", n_envs=n_envs) for _ in range(n_envs)]
    else:
        raise NotImplementedError("Only MovingDot and Hopper are supported")

    if use_async:
        envs = gym.vector.AsyncVectorEnv(ienvs)
    else:
        envs = gym.vector.SyncVectorEnv(ienvs)

    envs.env_name = env_name  # type: ignore
    return envs


def get_env_name(env: Union[gym.vector.AsyncVectorEnv, gym.vector.SyncVectorEnv]) -> str:
    if isinstance(env, gym.vector.AsyncVectorEnv) or isinstance(env, gym.vector.SyncVectorEnv):
        return env.envs[0].spec.id  # type: ignore
    else:
        return env.spec.id
