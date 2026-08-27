"""Tests for Booster T1 tracking task registration/configuration."""

from __future__ import annotations

import mjlab_playground  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab_playground.tasks.tracking.config.t1.env_cfgs import (
  booster_t1_flat_tracking_env_cfg,
)


def test_tracking_task_registered() -> None:
  assert "Mjlab-Tracking-Flat-Booster-T1" in list_tasks()


def test_tracking_task_configs_load() -> None:
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Booster-T1")
  rl_cfg = load_rl_cfg("Mjlab-Tracking-Flat-Booster-T1")
  assert "motion" in env_cfg.commands
  assert env_cfg.commands["motion"].anchor_body_name == "Trunk"
  assert env_cfg.commands["motion"].body_names[0] == "Trunk"
  assert "base_ang_vel_exceed" in env_cfg.terminations
  assert rl_cfg.experiment_name == "t1_tracking"


def test_tracking_play_cfg_overrides() -> None:
  cfg = booster_t1_flat_tracking_env_cfg(play=True)
  motion = cfg.commands["motion"]
  assert cfg.episode_length_s == int(1e9)
  assert "push_robot" not in cfg.events
  assert motion.pose_range == {}
  assert motion.velocity_range == {}
  assert motion.sampling_mode == "start"


def test_tracking_actor_terms_without_state_estimation() -> None:
  cfg = booster_t1_flat_tracking_env_cfg(has_state_estimation=False)
  actor_terms = cfg.observations["actor"].terms
  assert "motion_anchor_pos_b" not in actor_terms
  assert "base_lin_vel" not in actor_terms
  assert "motion_anchor_ori_b" in actor_terms
  assert "base_ang_vel" in actor_terms
