"""Booster K1 target-location AMP environment configuration."""

from __future__ import annotations

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab_playground.asset_zoo.robots.booster_k1.k1_constants import (
  get_k1_robot_cfg,
)
from mjlab_playground.tasks.target_location_amp import mdp as target_mdp
from mjlab_playground.tasks.target_location_amp.target_location_amp_env_cfg import (
  make_target_location_amp_env_cfg,
)

_K1_TARGET_POSITION_TOLERANCE = 0.16
_K1_TARGET_YAW_TOLERANCE = 0.20
_K1_TARGET_STABLE_HOLD_STEPS = 25
_K1_TARGET_MAX_LIN_SPEED = 0.35
_K1_TARGET_MAX_YAW_RATE = 0.60
_K1_TARGET_MIN_UP_Z = 0.75
_K1_EPISODE_LENGTH_S = 6.0
_K1_FIXED_TARGET_RESAMPLING_TIME_RANGE = (1.0e9, 1.0e9)

_K1_TARGET_LOCO_ACTION_SCALE = {
  "AAHead_yaw": 0.5,
  "Head_pitch": 0.5,
  "ALeft_Shoulder_Pitch": 1.0,
  "Left_Shoulder_Roll": 1.0,
  "Left_Elbow_Pitch": 1.0,
  "Left_Elbow_Yaw": 1.0,
  "ARight_Shoulder_Pitch": 1.0,
  "Right_Shoulder_Roll": 1.0,
  "Right_Elbow_Pitch": 1.0,
  "Right_Elbow_Yaw": 1.0,
  ".*_Hip_Pitch": 0.9,
  ".*_Hip_Roll": 0.8,
  ".*_Hip_Yaw": 0.8,
  ".*_Knee_Pitch": 1.0,
  ".*_Ankle_Pitch": 0.6,
  ".*_Ankle_Roll": 0.45,
}

_K1_TARGET_LOCO_GOAL_CURRICULUM_STAGES = [
  {
    "step": 0,
    "target_radius": (0.4, 1.2),
    "target_yaw": (-0.25, 0.25),
    "target_angle": (-0.50, 0.50),
    "resampling_time_range": None,
    "resample_on_near_goal": False,
    "resample_position_tolerance": _K1_TARGET_POSITION_TOLERANCE,
    "resample_yaw_tolerance": _K1_TARGET_YAW_TOLERANCE,
  },
  {
    "step": 600 * 24,
    "target_radius": (0.4, 2.4),
    "target_yaw": (-0.75, 0.75),
    "target_angle": (-1.25, 1.25),
    "resampling_time_range": None,
    "resample_on_near_goal": False,
    "resample_position_tolerance": _K1_TARGET_POSITION_TOLERANCE,
    "resample_yaw_tolerance": _K1_TARGET_YAW_TOLERANCE,
  },
  {
    "step": 1_600 * 24,
    "target_radius": (0.4, 4.0),
    "target_yaw": (-1.50, 1.50),
    "target_angle": (-2.20, 2.20),
    "resampling_time_range": None,
    "resample_on_near_goal": False,
    "resample_position_tolerance": _K1_TARGET_POSITION_TOLERANCE,
    "resample_yaw_tolerance": _K1_TARGET_YAW_TOLERANCE,
  },
  {
    "step": 3_500 * 24,
    "target_radius": (0.4, 6.0),
    "target_yaw": (-math.pi, math.pi),
    "target_angle": (-math.pi, math.pi),
    "resampling_time_range": None,
    "resample_on_near_goal": False,
    "resample_position_tolerance": _K1_TARGET_POSITION_TOLERANCE,
    "resample_yaw_tolerance": _K1_TARGET_YAW_TOLERANCE,
  },
]


def _apply_k1_joint_action_setup(cfg: ManagerBasedRlEnvCfg) -> None:
  joint_action = cfg.actions["joint_pos"]
  assert isinstance(joint_action, RelativeJointPositionActionCfg)
  joint_action.scale = _K1_TARGET_LOCO_ACTION_SCALE
  joint_action.clip = None


def _robot_asset_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=(".*",))


def _configure_k1_goal_reached_termination(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.terminations["goal_reached"] = TerminationTermCfg(
    func=target_mdp.goal_pose_reached,
    params={
      "command_name": "goal_pose",
      "position_tolerance": _K1_TARGET_POSITION_TOLERANCE,
      "yaw_tolerance": _K1_TARGET_YAW_TOLERANCE,
      "hold_steps": _K1_TARGET_STABLE_HOLD_STEPS,
      "entity_name": "robot",
      "max_lin_speed": _K1_TARGET_MAX_LIN_SPEED,
      "max_yaw_rate": _K1_TARGET_MAX_YAW_RATE,
      "min_up_z": _K1_TARGET_MIN_UP_Z,
    },
  )


def _configure_k1_goal_pose_command(cfg: ManagerBasedRlEnvCfg) -> None:
  goal_pose_cmd = cfg.commands["goal_pose"]
  assert isinstance(goal_pose_cmd, target_mdp.GoalPoseCommandCfg)
  stage0 = _K1_TARGET_LOCO_GOAL_CURRICULUM_STAGES[0]
  goal_pose_cmd.ranges.target_radius = stage0["target_radius"]
  goal_pose_cmd.ranges.target_yaw = stage0["target_yaw"]
  goal_pose_cmd.ranges.target_angle = stage0["target_angle"]
  goal_pose_cmd.resampling_time_range = _K1_FIXED_TARGET_RESAMPLING_TIME_RANGE
  goal_pose_cmd.resample_on_near_goal = stage0["resample_on_near_goal"]
  goal_pose_cmd.resample_position_tolerance = stage0["resample_position_tolerance"]
  goal_pose_cmd.resample_yaw_tolerance = stage0["resample_yaw_tolerance"]


def booster_k1_target_location_amp_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the K1 target-location locomotion AMP task."""
  cfg = make_target_location_amp_env_cfg(play=play)
  cfg.scene.entities = {
    "robot": deepcopy(get_k1_robot_cfg()),
  }
  cfg.scene.num_envs = 1 if play else 1024
  cfg.episode_length_s = _K1_EPISODE_LENGTH_S
  cfg.viewer.body_name = "torso"
  _apply_k1_joint_action_setup(cfg)
  _configure_k1_goal_pose_command(cfg)
  cfg.events["reset_robot_joints"].params["asset_cfg"] = _robot_asset_cfg()
  cfg.observations["critic"].terms["amp_robot"].params["robot_cfg"] = _robot_asset_cfg()

  if not play:
    _configure_k1_goal_reached_termination(cfg)
    cfg.curriculum["goal_pose"] = CurriculumTermCfg(
      func=target_mdp.goal_pose_ranges,
      params={
        "command_name": "goal_pose",
        "stages": _K1_TARGET_LOCO_GOAL_CURRICULUM_STAGES,
      },
    )

  if play:
    cfg.observations["actor"].enable_corruption = False
    goal_pose_cmd = cfg.commands["goal_pose"]
    assert isinstance(goal_pose_cmd, target_mdp.GoalPoseCommandCfg)
    goal_pose_cmd.ranges.target_radius = (0.4, 6.0)
    goal_pose_cmd.ranges.target_yaw = (-math.pi, math.pi)
    goal_pose_cmd.ranges.target_angle = (-math.pi, math.pi)
    goal_pose_cmd.resampling_time_range = _K1_FIXED_TARGET_RESAMPLING_TIME_RANGE
    goal_pose_cmd.resample_on_near_goal = False
    goal_pose_cmd.resample_position_tolerance = _K1_TARGET_POSITION_TOLERANCE
    goal_pose_cmd.resample_yaw_tolerance = _K1_TARGET_YAW_TOLERANCE

  return cfg
