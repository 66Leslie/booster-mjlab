"""RL configuration for Booster K1 target-location AMP locomotion."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg

from mjlab_playground.amp.rl.motion_loader import (
  default_k1_locomotion_amp_motion_groups,
)
from mjlab_playground.amp.rl.runner_cfg import (
  AmpOnPolicyRunnerCfg,
  AmpPpoAlgorithmCfg,
)

_STABLE_ACTION_DISTRIBUTION = {
  "class_name": "mjlab_playground.amp.rl.distributions:ClampedGaussianDistribution",
  "init_std": 0.45,
  "std_type": "log",
  "min_std": 0.05,
  "max_std": 1.0,
}


def booster_k1_target_location_amp_runner_cfg() -> AmpOnPolicyRunnerCfg:
  """Create RL runner configuration for K1 target-location locomotion AMP."""
  return AmpOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=dict(_STABLE_ACTION_DISTRIBUTION),
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=AmpPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.001,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.98,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      amp_cfg={
        "motion_groups": default_k1_locomotion_amp_motion_groups(),
        "target_fps": 50.0,
        "expected_dof": 22,
        "reward_coef": 0.6,
        "reward_scale": 1.0,
        "task_reward_lerp": 0.45,
        "loss_coef": 1.0,
        "grad_penalty_coef": 10.0,
        "discriminator_learning_rate": 3.0e-4,
        "replay_buffer_size": 100_000,
        "discriminator_hidden_dims": (512, 256),
        "discriminator_activation": "relu",
        "normalize_input": True,
        "discriminator_loss_type": "least_squares",
        "discriminator_reward_type": "hardware_lsgan",
        "discriminator_expert_label": 0.9,
        "discriminator_policy_label": 0.1,
        "discriminator_reward_clip_max": 1.5,
      },
      symmetry_cfg=None,
    ),
    experiment_name="k1_target_location_amp",
    run_name="k1_target_location_amp",
    wandb_project="mjlab_playground",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
    clip_actions=1.2,
  )
