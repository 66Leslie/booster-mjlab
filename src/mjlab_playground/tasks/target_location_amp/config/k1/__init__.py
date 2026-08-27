"""Booster K1 target-location AMP task registration."""

from mjlab.tasks.registry import register_mjlab_task

from mjlab_playground.amp.rl import AmpOnPolicyRunner

from .env_cfgs import booster_k1_target_location_amp_flat_env_cfg
from .rl_cfg import booster_k1_target_location_amp_runner_cfg

register_mjlab_task(
  task_id="Mjlab-TargetLocationAmp-Flat-Booster-K1",
  env_cfg=booster_k1_target_location_amp_flat_env_cfg(),
  play_env_cfg=booster_k1_target_location_amp_flat_env_cfg(play=True),
  rl_cfg=booster_k1_target_location_amp_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)
