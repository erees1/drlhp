"""
A simple OpenAI gym environment consisting of a white dot moving in a black
square.

# Taken from: https://github.com/mrahtz/gym-moving-dot/blob/master/gym_moving_dot/envs/moving_dot_env.py
"""

from typing import Any, Optional, TypeVar
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
import pygame


class ALE(object):
    def __init__(self):
        self.lives = lambda: 0


ActType = TypeVar("ActType")


class MovingDotEnv(gym.Env[NDArray[np.uint8], ActType]):  # type: ignore
    """Base class for MovingDot game"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 150}

    def __init__(
        self,
        render_mode="human",
        channel_dim: bool = True,
        size: tuple[int, int] = (210, 160),
        max_steps=1000,
        random_start=True,
    ):
        super(
            gym.Env,
            self,
        ).__init__()

        # Environment parameters
        self.dot_size = [4, 4]
        self.max_steps = max_steps
        self.random_start = random_start

        self.channel_dim = channel_dim
        self.size = size
        # environment setup
        if not channel_dim:
            self.observation_space = spaces.Box(low=0, high=255, shape=(size[0], size[1]), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)

        self.centre = np.array([size[1] // 2, size[0] // 2])

        # Needed by atari_wrappers in OpenAI baselines
        self.ale = ALE()

        # Rendering
        self.clock = None
        self.window_size = 300
        self.window = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        if self.random_start:
            x = self.np_random.integers(low=0, high=self.size[1])
            y = self.np_random.integers(low=0, high=self.size[0])
            self.pos = [x, y]
        else:
            self.pos = [0, 0]
        self.steps = 0
        ob = self._get_ob()

        info: dict[str, Any] = {}
        return ob, info

    def _get_ob(self) -> NDArray[np.uint8]:
        ob = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
        if self.channel_dim:
            ob = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
        else:
            ob = np.zeros(self.size, dtype=np.uint8)

        x = self.pos[0]
        y = self.pos[1]
        w = self.dot_size[0]
        h = self.dot_size[1]
        if self.channel_dim:
            ob[y - h : y + h, x - w : x + w, :] = 255
        else:
            ob[y - h : y + h, x - w : x + w] = 255
        return ob

    def _render_frame(self) -> Optional[NDArray[np.uint8]]:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))

        # Use ob to fill the canvas.
        ob = self._get_ob()
        # resize ob to fit the canvas
        ob = cv2.resize(ob, (self.window_size, self.window_size), interpolation=cv2.INTER_NEAREST)
        pygame_surface = pygame.surfarray.make_surface(ob)
        canvas.blit(pygame_surface, (0, 0))

        if self.render_mode == "human":
            assert self.window is not None
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def get_action_meanings(self):
        return ["NOOP", "DOWN", "RIGHT", "UP", "LEFT"]

    def step(self, action: ActType) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:
        prev_pos = self.pos[:]

        self._update_pos(action)

        ob = self._get_ob()

        self.steps += 1
        if self.steps < self.max_steps:
            episode_over = False
        else:
            episode_over = True

        dist1 = np.linalg.norm(prev_pos - self.centre)
        dist2 = np.linalg.norm(self.pos - self.centre)
        if dist2 < dist1:
            reward = 1
        elif dist2 == dist1:
            reward = 0
        else:
            reward = -1

        info: dict[str, Any] = {}
        return ob, reward, episode_over, False, info

    def _update_pos(self, action: ActType) -> None:
        """subclass is supposed to implement the logic
        to update the frame given an action at t"""
        raise NotImplementedError

    # Based on gym's atari_env.py
    def render(self, mode="human", close=False) -> Optional[NDArray[np.uint8]]:
        if close:
            if self.window is not None:
                pygame.quit()
                self.window = None
            return

        # We only import this here in case we're running on a headless server
        # from gymnasium.utils import rendering
        assert mode == "human", "MovingDot only supports human render mode"
        return self._render_frame()


class MovingDotDiscreteEnv(MovingDotEnv[np.int64]):
    """Discrete Action MovingDot env"""

    def __init__(
        self,
        render_mode="human",
        channel_dim: bool = True,
        size: tuple[int, int] = (210, 160),
        max_steps=1000,
        random_start=True,
    ):
        super(MovingDotDiscreteEnv, self).__init__(
            render_mode=render_mode, channel_dim=channel_dim, size=size, max_steps=max_steps, random_start=random_start
        )
        self.action_space: spaces.Space[np.int64] = spaces.Discrete(5)

    def _update_pos(self, action: np.int64):
        assert action >= 0 and action <= 4

        if action == 0:
            # NOOP
            pass
        elif action == 1:
            self.pos[1] += 1
        elif action == 2:
            self.pos[0] += 1
        elif action == 3:
            self.pos[1] -= 1
        elif action == 4:
            self.pos[0] -= 1
        self.pos[0] = np.clip(self.pos[0], self.dot_size[0], 159 - self.dot_size[0])
        self.pos[1] = np.clip(self.pos[1], self.dot_size[1], 209 - self.dot_size[1])


class MovingDotContinuousEnv(MovingDotEnv[NDArray[np.float32]]):
    """Continuous Action MovingDot env"""

    def __init__(
        self,
        low=-1,
        high=1,
        moving_thd=0.1,
        render_mode="human",
        channel_dim: bool = True,
        size: tuple[int, int] = (210, 160),
        max_steps=1000,
        random_start=True,
    ):  # moving_thd is empirically determined
        super(MovingDotContinuousEnv, self).__init__(
            render_mode=render_mode, channel_dim=channel_dim, size=size, max_steps=max_steps, random_start=random_start
        )

        self._high = high
        self._low = low
        self._moving_thd = moving_thd  # used to decide if the object has to move, see step func below.
        self.action_space: spaces.Space[NDArray[np.float32]] = spaces.Box(
            low=low, high=high, shape=(2,), dtype=np.float32
        )

    def _update_pos(self, action: NDArray[np.float32]):
        _x, _y = action
        assert self._low <= _x <= self._high, "movement along x-axis has to fall in between -1 to 1"
        assert self._low <= _y <= self._high, "movement along y-axis has to fall in between -1 to 1"

        """
        [Note]
        Since the action values are continuous for each x/y pos,
        we round the position of the object after executing the action on the 2D space.
        """
        new_x = self.pos[0] + 1 if _x >= self._moving_thd else self.pos[0] - 1
        new_y = self.pos[1] + 1 if _y >= self._moving_thd else self.pos[1] - 1

        self.pos[0] = np.clip(new_x, self.dot_size[0], 159 - self.dot_size[0])
        self.pos[1] = np.clip(new_y, self.dot_size[1], 209 - self.dot_size[1])
