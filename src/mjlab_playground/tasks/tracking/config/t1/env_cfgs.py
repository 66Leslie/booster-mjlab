"""Booster T1 tracking environment configuration."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, TerminationTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from mjlab_playground.asset_zoo.robots.booster_t1.t1_constants import (
  T1_ACTION_SCALE,
  get_t1_robot_cfg,
)
from mjlab_playground.tasks.tracking.mdp.commands import (
  TrackingMixedResetMotionCommandCfg,
)
from mjlab_playground.tasks.tracking.mdp.events import (
  TRACKING_HOME_RESET_MASK_KEY,
  sample_tracking_reset_source,
)

_MAX_ANG_VEL = 500 * math.pi / 180.0  # [rad/s]
_HOME_RESET_PROBABILITY_TRAIN = 0.35
_HOME_RESET_PROBABILITY_PLAY = 1.0
_HOME_JOINT_VEL_RANGE = (0.0, 0.0)
_HOME_HOLD_STEPS_TRAIN = 0
_HOME_HOLD_STEPS_PLAY = 20

_T1_TRACKING_BODY_NAMES = (
  # MotionCommand reset treats body_names[0] as the floating base body.
  "Trunk",
  "Waist",
  "Hip_Roll_Left",
  "Shank_Left",
  "left_foot_link",
  "Hip_Roll_Right",
  "Shank_Right",
  "right_foot_link",
  "AL1",
  "AL3",
  "left_hand_link",
  "AR1",
  "AR3",
  "right_hand_link",
)


def _resolve_default_tracking_motion_file() -> str:
  asset_motion_root = (
    Path(__file__).resolve().parents[3]
    / "asset_zoo"
    / "motions"
    / "retargeted"
    / "tracking"
    / "booster_t1_23dof"
  )
  candidates = [
    os.getenv("MJLAB_TRACKING_MOTION_FILE", ""),
    str(asset_motion_root / "0026_kicking2_stageii.booster_t1_23dof.motion.npz"),
  ]
  for candidate in candidates:
    if candidate and Path(candidate).exists():
      return candidate
  return candidates[1]


_TRACKING_MOTION_FILE = _resolve_default_tracking_motion_file()


def base_ang_vel_exceed(
  env: ManagerBasedRlEnv,
  threshold: float,
) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  ang_vel = asset.data.root_link_ang_vel_b
  return torch.any(ang_vel.abs() > threshold, dim=-1)


def booster_t1_flat_tracking_env_cfg(
  has_state_estimation: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 flat tracking configuration."""
  if _T1_TRACKING_BODY_NAMES[0] != "Trunk":
    raise ValueError(
      "Tracking body_names must start with 'Trunk' for Booster T1. "
      "MotionCommand reset uses body_names[0] as root state."
    )

  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": get_t1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = T1_ACTION_SCALE

  motion_cmd_base = cfg.commands["motion"]
  assert isinstance(motion_cmd_base, MotionCommandCfg)
  motion_cmd = TrackingMixedResetMotionCommandCfg(
    entity_name=motion_cmd_base.entity_name,
    resampling_time_range=motion_cmd_base.resampling_time_range,
    debug_vis=motion_cmd_base.debug_vis,
    pose_range=dict(motion_cmd_base.pose_range),
    velocity_range=dict(motion_cmd_base.velocity_range),
    joint_position_range=motion_cmd_base.joint_position_range,
    adaptive_kernel_size=motion_cmd_base.adaptive_kernel_size,
    adaptive_lambda=motion_cmd_base.adaptive_lambda,
    adaptive_uniform_ratio=motion_cmd_base.adaptive_uniform_ratio,
    adaptive_alpha=motion_cmd_base.adaptive_alpha,
    sampling_mode=motion_cmd_base.sampling_mode,
    motion_file=motion_cmd_base.motion_file,
    anchor_body_name=motion_cmd_base.anchor_body_name,
    body_names=motion_cmd_base.body_names,
    viz=copy.deepcopy(motion_cmd_base.viz),
    home_start_frame=0,
    home_pose_range={},
    home_velocity_range={},
    home_joint_position_range=(0.0, 0.0),
    home_joint_velocity_range=_HOME_JOINT_VEL_RANGE,
    home_root_position_mode="motion_xyz",
    home_root_orientation_mode="motion_yaw",
    home_canonicalize_motion_yaw=False,
    home_initial_hold_steps=_HOME_HOLD_STEPS_TRAIN,
  )
  cfg.commands["motion"] = motion_cmd
  motion_cmd.motion_file = _TRACKING_MOTION_FILE
  motion_cmd.anchor_body_name = "Trunk"
  motion_cmd.body_names = _T1_TRACKING_BODY_NAMES

  cfg.events["tracking_reset_source"] = EventTermCfg(
    func=sample_tracking_reset_source,
    mode="reset",
    params={
      "home_probability": _HOME_RESET_PROBABILITY_TRAIN,
      "mask_key": TRACKING_HOME_RESET_MASK_KEY,
    },
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-4]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_foot_link",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
  )
  cfg.terminations["base_ang_vel_exceed"] = TerminationTermCfg(
    func=base_ang_vel_exceed,
    params={"threshold": _MAX_ANG_VEL},
  )
  cfg.viewer.body_name = "Trunk"

  # Remove state-estimation terms for pure proprio + motion command tracking.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events["tracking_reset_source"].params["home_probability"] = (
      _HOME_RESET_PROBABILITY_PLAY
    )

    # Disable RSI randomization in play mode.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
    # Match competition kickoff intuition: stand/home orientation, no forced
    # leaning root pitch/roll from motion frame, and give a short warmup window
    # before advancing the reference timeline.
    motion_cmd.home_root_position_mode = "motion_xy_home_z"
    motion_cmd.home_root_orientation_mode = "home"
    motion_cmd.home_canonicalize_motion_yaw = True
    motion_cmd.home_initial_hold_steps = _HOME_HOLD_STEPS_PLAY

  return cfg
