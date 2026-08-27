from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_go1_getup_env_cfg
from .rl_cfg import unitree_go1_getup_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Getup-Flat-Unitree-Go1",
  env_cfg=unitree_go1_getup_env_cfg(),
  play_env_cfg=unitree_go1_getup_env_cfg(play=True),
  rl_cfg=unitree_go1_getup_ppo_runner_cfg(),
)
