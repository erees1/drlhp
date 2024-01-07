from dataclasses import dataclass


@dataclass(frozen=True)
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
    steps_per_update_per_env: int = 640
    n_timesteps: int = int(1e6)
    optimization_epochs: int = 2
    batch_size: int = 32
    log_every: int = 100
    entropy_coeff: float = 0.01
    clip_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    render_every: int = 5
    n_envs: int = 4
    gamma: float = 0.99
    tau: float = 0.95
    grad_clip: float = 0.5
    use_async_env: bool = False
