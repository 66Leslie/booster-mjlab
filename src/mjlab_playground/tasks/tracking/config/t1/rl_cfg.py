"""RL configuration for Booster T1 tracking."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def booster_t1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Booster T1 tracking task."""
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "t1_tracking"
  cfg.run_name = "t1_tracking"
  cfg.wandb_project = "mjlab_playground"
  return cfg
