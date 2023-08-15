from gymnasium.envs.registration import register

register(id="MovingDotDiscreteNoFrameskip-v0", entry_point="drlhp.gym_moving_dot.moving_dot_env:MovingDotDiscreteEnv")

register(id="MovingDotDiscrete-v0", entry_point="drlhp.gym_moving_dot.moving_dot_env:MovingDotDiscreteEnv")

register(
    id="MovingDotContinuousNoFrameskip-v0", entry_point="drlhp.gym_moving_dot.moving_dot_env:MovingDotContinuousEnv"
)

register(id="MovingDotContinuous-v0", entry_point="drlhp.gym_moving_dot.moving_dot_env:MovingDotContinuousEnv")
