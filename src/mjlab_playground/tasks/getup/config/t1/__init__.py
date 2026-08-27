from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import booster_t1_getup_env_cfg
from .rl_cfg import booster_t1_getup_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Getup-Flat-Booster-T1",
  env_cfg=booster_t1_getup_env_cfg(),
  play_env_cfg=booster_t1_getup_env_cfg(play=True),
  rl_cfg=booster_t1_getup_ppo_runner_cfg(),
)
