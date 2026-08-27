"""Booster T1 target-location AMP task registration."""

from mjlab.tasks.registry import register_mjlab_task

from mjlab_playground.amp.rl import AmpOnPolicyRunner

from .env_cfgs import (
  booster_t1_target_location_amp_competition_collision_flat_env_cfg,
  booster_t1_target_location_amp_competition_foot_flat_env_cfg,
  booster_t1_target_location_amp_flat_env_cfg,
)
from .rl_cfg import (
  booster_t1_target_location_amp_competition_collision_runner_cfg,
  booster_t1_target_location_amp_competition_foot_runner_cfg,
  booster_t1_target_location_amp_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-TargetLocationAmp-Flat-Booster-T1",
  env_cfg=booster_t1_target_location_amp_flat_env_cfg(),
  play_env_cfg=booster_t1_target_location_amp_flat_env_cfg(play=True),
  rl_cfg=booster_t1_target_location_amp_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-TargetLocationAmp-CompetitionFoot-Flat-Booster-T1",
  env_cfg=booster_t1_target_location_amp_competition_foot_flat_env_cfg(),
  play_env_cfg=booster_t1_target_location_amp_competition_foot_flat_env_cfg(play=True),
  rl_cfg=booster_t1_target_location_amp_competition_foot_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-TargetLocationAmp-CompetitionCollision-Flat-Booster-T1",
  env_cfg=booster_t1_target_location_amp_competition_collision_flat_env_cfg(),
  play_env_cfg=booster_t1_target_location_amp_competition_collision_flat_env_cfg(
    play=True
  ),
  rl_cfg=booster_t1_target_location_amp_competition_collision_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)
