"""Booster T1 target-location AMP environment configuration."""

from __future__ import annotations

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_playground.amp.mdp.actions import DelayedJointPositionActionCfg
from mjlab_playground.asset_zoo.robots.booster_t1.t1_constants import (
  T1_ACTION_SCALE,
  get_t1_competition_collision_robot_cfg,
  get_t1_competition_foot_robot_cfg,
  get_t1_robot_cfg,
)
from mjlab_playground.tasks.target_location_amp import mdp as target_mdp
from mjlab_playground.tasks.target_location_amp.target_location_amp_env_cfg import (
  make_target_location_amp_env_cfg,
)

_T1_TARGET_POSITION_TOLERANCE = 0.22
_T1_TARGET_YAW_TOLERANCE = 0.30
_T1_NEAR_GOAL_DISTANCE = 0.85
_T1_REWARD_GATE_MIN_ROOT_HEIGHT = 0.58
_T1_REWARD_GATE_MIN_UP_Z = 0.80
_T1_FALL_HEIGHT_THRESHOLD = 0.50
_T1_FALL_UP_Z_THRESHOLD = 0.70
_T1_EPISODE_LENGTH_S = 20.0
_T1_TARGET_STABLE_HOLD_STEPS = 25
_T1_TARGET_MAX_LIN_SPEED = 0.35
_T1_TARGET_MAX_YAW_RATE = 0.60
_T1_TARGET_MIN_UP_Z = 0.75
_T1_RL_STEPS_PER_ITERATION = 24
_T1_COMPETITION_ACTION_DELAY_STEPS = 1
_T1_FINE_TUNE_ITERATION = 70_500
_T1_FINE_TARGET_POSITION_TOLERANCE = 0.08
_T1_FINE_TARGET_YAW_TOLERANCE = math.radians(6.0)
_T1_FINE_NEAR_GOAL_DISTANCE = 0.45
_T1_FINE_TARGET_STABLE_HOLD_STEPS = 25
_T1_FINE_TARGET_MAX_LIN_SPEED = 0.25
_T1_FINE_TARGET_MAX_YAW_RATE = 0.35
_T1_FINE_TARGET_MIN_UP_Z = 0.80
_T1_FINE_NEAR_GOAL_STABILITY_WEIGHT = 0.50
_T1_FINE_GOAL_POSITION_BONUS_WEIGHT = 0.50
_T1_FINE_GOAL_POSE_BONUS_WEIGHT = 1.00
_T1_FINE_JOINT_TARGET_LIMITS_WEIGHT = -0.03
_T1_FOOT_FRICTION_GEOMS = r"^(left|right)_foot[1-4]_collision$"
_T1_COMPETITION_FOOT_FRICTION_GEOMS = r"^(left|right)_foot$"
_T1_FINE_TARGET_COMMAND_BUCKETS = [
  {
    "radius_range": (0.0, 0.0),
    "yaw_range": (0.0, 0.0),
    "angle_range": (0.0, 0.0),
    "probability": 0.1,
  },
  {
    "radius_range": (0.0, 0.0),
    "yaw_range": (-math.pi, math.pi),
    "angle_range": (0.0, 0.0),
    "probability": 0.1,
  },
  {
    "radius_range": (0.0, 2.0),
    "yaw_range": None,
    "angle_range": None,
    "probability": 0.3,
  },
  {
    "radius_range": (5.0, 6.0),
    "yaw_range": None,
    "angle_range": None,
    "probability": 0.4,
  },
  {
    "radius_range": (2.0, 4.0),
    "yaw_range": None,
    "angle_range": None,
    "probability": 0.1,
  },
]
_T1_TARGET_LOCO_GOAL_CURRICULUM_STAGES = [
  {
    "step": 0,
    "target_radius": (0.3, 0.8),
    "target_yaw": (-0.2, 0.2),
    "target_angle": (-0.4, 0.4),
    "resampling_time_range": (4.0, 8.0),
    "resample_on_near_goal": True,
    "resample_on_goal_success": False,
    "resample_position_tolerance": _T1_TARGET_POSITION_TOLERANCE,
    "resample_yaw_tolerance": _T1_TARGET_YAW_TOLERANCE,
  },
  {
    "step": 2_000 * _T1_RL_STEPS_PER_ITERATION,
    "target_radius": (0.3, 1.5),
    "target_yaw": (-0.5, 0.5),
    "target_angle": (-1.0, 1.0),
    "resampling_time_range": (4.0, 8.0),
    "resample_on_near_goal": True,
    "resample_on_goal_success": False,
    "resample_position_tolerance": _T1_TARGET_POSITION_TOLERANCE,
    "resample_yaw_tolerance": _T1_TARGET_YAW_TOLERANCE,
  },
  {
    "step": 4_000 * _T1_RL_STEPS_PER_ITERATION,
    "target_radius": (0.5, 3.0),
    "target_yaw": (-1.0, 1.0),
    "target_angle": (-math.pi, math.pi),
    "resampling_time_range": (4.0, 8.0),
    "resample_on_near_goal": True,
    "resample_on_goal_success": False,
    "resample_position_tolerance": _T1_TARGET_POSITION_TOLERANCE,
    "resample_yaw_tolerance": _T1_TARGET_YAW_TOLERANCE,
  },
  {
    "step": 7_000 * _T1_RL_STEPS_PER_ITERATION,
    "target_radius": (0.5, 5.0),
    "target_yaw": (-math.pi, math.pi),
    "target_angle": (-math.pi, math.pi),
    "resampling_time_range": (4.0, 8.0),
    "resample_on_near_goal": True,
    "resample_on_goal_success": False,
    "resample_position_tolerance": _T1_TARGET_POSITION_TOLERANCE,
    "resample_yaw_tolerance": _T1_TARGET_YAW_TOLERANCE,
  },
  {
    "step": 24_000 * _T1_RL_STEPS_PER_ITERATION,
    "target_radius": (1.0, 7.0),
    "target_yaw": (-math.pi, math.pi),
    "target_angle": (-math.pi, math.pi),
    "resampling_time_range": (5.0, 10.0),
    "resample_on_near_goal": True,
    "resample_on_goal_success": False,
    "resample_position_tolerance": _T1_TARGET_POSITION_TOLERANCE,
    "resample_yaw_tolerance": _T1_TARGET_YAW_TOLERANCE,
  },
  {
    "step": 27_000 * _T1_RL_STEPS_PER_ITERATION,
    "target_radius": (1.0, 10.0),
    "target_yaw": (-math.pi, math.pi),
    "target_angle": (-math.pi, math.pi),
    "resampling_time_range": (2.0, 12.0),
    "resample_on_near_goal": True,
    "resample_on_goal_success": False,
    "resample_position_tolerance": _T1_TARGET_POSITION_TOLERANCE,
    "resample_yaw_tolerance": _T1_TARGET_YAW_TOLERANCE,
  },
  {
    "step": _T1_FINE_TUNE_ITERATION * _T1_RL_STEPS_PER_ITERATION,
    "target_radius": (0.0, 6.0),
    "target_command_buckets": _T1_FINE_TARGET_COMMAND_BUCKETS,
    "target_yaw": (-math.pi, math.pi),
    "target_angle": (-math.pi, math.pi),
    "resampling_time_range": (1.0, 4.0),
    "resample_on_near_goal": False,
    "resample_on_goal_success": True,
    "resample_position_tolerance": _T1_FINE_TARGET_POSITION_TOLERANCE,
    "resample_yaw_tolerance": _T1_FINE_TARGET_YAW_TOLERANCE,
    "goal_position_tolerance": _T1_FINE_TARGET_POSITION_TOLERANCE,
    "goal_yaw_tolerance": _T1_FINE_TARGET_YAW_TOLERANCE,
    "goal_hold_steps": _T1_FINE_TARGET_STABLE_HOLD_STEPS,
    "goal_max_lin_speed": _T1_FINE_TARGET_MAX_LIN_SPEED,
    "goal_max_yaw_rate": _T1_FINE_TARGET_MAX_YAW_RATE,
    "goal_min_up_z": _T1_FINE_TARGET_MIN_UP_Z,
    "near_goal_heading_distance": _T1_FINE_NEAR_GOAL_DISTANCE,
    "near_goal_stability_weight": _T1_FINE_NEAR_GOAL_STABILITY_WEIGHT,
    "near_goal_stability_distance": _T1_FINE_NEAR_GOAL_DISTANCE,
    "near_goal_stability_lin_speed_std": 0.30,
    "near_goal_stability_yaw_rate_std": 0.45,
    "near_goal_stability_min_up_z": _T1_FINE_TARGET_MIN_UP_Z,
    "goal_pose_bonus_weight": _T1_FINE_GOAL_POSE_BONUS_WEIGHT,
    "joint_target_limits_weight": _T1_FINE_JOINT_TARGET_LIMITS_WEIGHT,
  },
]


def _robot_asset_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=(".*",))


def _apply_t1_joint_action_setup(
  cfg: ManagerBasedRlEnvCfg,
  *,
  action_delay_steps: int = 0,
) -> None:
  action_cfg_type = (
    DelayedJointPositionActionCfg if action_delay_steps > 0 else JointPositionActionCfg
  )
  action_cfg_kwargs = {}
  if action_delay_steps > 0:
    action_cfg_kwargs["delay_steps"] = action_delay_steps
  cfg.actions["joint_pos"] = action_cfg_type(
    entity_name="robot",
    actuator_names=(".*",),
    scale=T1_ACTION_SCALE,
    clip=None,
    use_default_offset=True,
    **action_cfg_kwargs,
  )


def _configure_t1_domain_randomization(
  cfg: ManagerBasedRlEnvCfg,
  *,
  foot_geom_names: str = _T1_FOOT_FRICTION_GEOMS,
) -> None:
  cfg.events["push_robot"] = EventTermCfg(
    func=env_mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={
      "velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
      },
    },
  )
  cfg.events["foot_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        geom_names=foot_geom_names,
      ),
      "operation": "abs",
      "ranges": (0.3, 1.2),
      "shared_random": True,
    },
  )
  cfg.events["encoder_bias"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "bias_range": (-0.015, 0.015),
    },
  )
  cfg.events["base_com"] = EventTermCfg(
    mode="startup",
    func=dr.body_com_offset,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
      "operation": "add",
      "ranges": {
        0: (-0.025, 0.025),
        1: (-0.025, 0.025),
        2: (-0.03, 0.03),
      },
    },
  )


def _configure_t1_goal_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.rewards["near_goal_stability"].weight = 0.0
  cfg.rewards["near_goal_stability"].params["near_distance"] = 0.75
  cfg.rewards["near_goal_stability"].params["position_tolerance"] = (
    _T1_FINE_TARGET_POSITION_TOLERANCE
  )
  cfg.rewards["near_goal_stability"].params["yaw_tolerance"] = (
    _T1_FINE_TARGET_YAW_TOLERANCE
  )
  cfg.rewards["near_goal_stability"].params["lin_speed_std"] = 0.45
  cfg.rewards["near_goal_stability"].params["yaw_rate_std"] = 0.75
  cfg.rewards["near_goal_stability"].params["min_up_z"] = 0.75

  cfg.rewards["position_progress"].params["position_tolerance"] = (
    _T1_TARGET_POSITION_TOLERANCE
  )
  cfg.rewards["position_progress"].params["yaw_tolerance"] = _T1_TARGET_YAW_TOLERANCE
  cfg.rewards["position_progress"].params["entity_name"] = "robot"
  cfg.rewards["position_progress"].params["min_root_height"] = (
    _T1_REWARD_GATE_MIN_ROOT_HEIGHT
  )
  cfg.rewards["position_progress"].params["min_up_z"] = _T1_REWARD_GATE_MIN_UP_Z
  cfg.rewards["near_goal_heading_alignment"].weight = 0.70
  cfg.rewards["near_goal_heading_alignment"].params["near_distance"] = (
    _T1_NEAR_GOAL_DISTANCE
  )
  cfg.rewards["near_goal_heading_alignment"].params["entity_name"] = "robot"
  cfg.rewards["near_goal_heading_alignment"].params["min_root_height"] = (
    _T1_REWARD_GATE_MIN_ROOT_HEIGHT
  )
  cfg.rewards["near_goal_heading_alignment"].params["min_up_z"] = (
    _T1_REWARD_GATE_MIN_UP_Z
  )
  cfg.rewards["yaw_progress"].params["position_tolerance"] = (
    _T1_TARGET_POSITION_TOLERANCE
  )
  cfg.rewards["yaw_progress"].params["yaw_tolerance"] = _T1_TARGET_YAW_TOLERANCE
  cfg.rewards["yaw_progress"].params["near_distance"] = _T1_NEAR_GOAL_DISTANCE
  cfg.rewards["yaw_progress"].params["entity_name"] = "robot"
  cfg.rewards["yaw_progress"].params["min_root_height"] = (
    _T1_REWARD_GATE_MIN_ROOT_HEIGHT
  )
  cfg.rewards["yaw_progress"].params["min_up_z"] = _T1_REWARD_GATE_MIN_UP_Z
  cfg.rewards["goal_position_bonus"].params["position_tolerance"] = (
    _T1_TARGET_POSITION_TOLERANCE
  )
  cfg.rewards["goal_position_bonus"].weight = _T1_FINE_GOAL_POSITION_BONUS_WEIGHT
  cfg.rewards["goal_position_bonus"].params["yaw_soft_tolerance"] = (
    _T1_TARGET_YAW_TOLERANCE
  )
  cfg.rewards["goal_position_bonus"].params["entity_name"] = "robot"
  cfg.rewards["goal_position_bonus"].params["min_root_height"] = (
    _T1_REWARD_GATE_MIN_ROOT_HEIGHT
  )
  cfg.rewards["goal_position_bonus"].params["min_up_z"] = _T1_REWARD_GATE_MIN_UP_Z
  cfg.rewards["goal_pose_bonus"].weight = 0.0
  cfg.rewards["goal_pose_bonus"].params["position_tolerance"] = (
    _T1_TARGET_POSITION_TOLERANCE
  )
  cfg.rewards["goal_pose_bonus"].params["yaw_tolerance"] = _T1_TARGET_YAW_TOLERANCE
  cfg.rewards["goal_pose_bonus"].params["hold_steps"] = _T1_TARGET_STABLE_HOLD_STEPS
  cfg.rewards["goal_pose_bonus"].params["entity_name"] = "robot"
  cfg.rewards["goal_pose_bonus"].params["max_lin_speed"] = _T1_TARGET_MAX_LIN_SPEED
  cfg.rewards["goal_pose_bonus"].params["max_yaw_rate"] = _T1_TARGET_MAX_YAW_RATE
  cfg.rewards["goal_pose_bonus"].params["min_up_z"] = _T1_TARGET_MIN_UP_Z
  cfg.rewards["goal_time_efficiency"].params["position_tolerance"] = (
    _T1_TARGET_POSITION_TOLERANCE
  )
  cfg.rewards["goal_time_efficiency"].params["yaw_tolerance"] = _T1_TARGET_YAW_TOLERANCE
  cfg.rewards["goal_time_efficiency"].params["hold_steps"] = (
    _T1_TARGET_STABLE_HOLD_STEPS
  )
  cfg.rewards["goal_time_efficiency"].params["entity_name"] = "robot"
  cfg.rewards["goal_time_efficiency"].params["max_lin_speed"] = _T1_TARGET_MAX_LIN_SPEED
  cfg.rewards["goal_time_efficiency"].params["max_yaw_rate"] = _T1_TARGET_MAX_YAW_RATE
  cfg.rewards["goal_time_efficiency"].params["min_up_z"] = _T1_TARGET_MIN_UP_Z
  cfg.rewards["fall_penalty"].weight = -0.2
  cfg.rewards["fall_penalty"].params["height_threshold"] = _T1_FALL_HEIGHT_THRESHOLD
  cfg.rewards["fall_penalty"].params["up_z_threshold"] = _T1_FALL_UP_Z_THRESHOLD


def _apply_t1_goal_stage_to_reward_cfg(cfg: ManagerBasedRlEnvCfg, stage: dict) -> None:
  if stage.get("goal_position_tolerance") is not None:
    position_tolerance = stage["goal_position_tolerance"]
    cfg.rewards["position_progress"].params["position_tolerance"] = position_tolerance
    cfg.rewards["yaw_progress"].params["position_tolerance"] = position_tolerance
    cfg.rewards["goal_position_bonus"].params["position_tolerance"] = position_tolerance
    cfg.rewards["goal_pose_bonus"].params["position_tolerance"] = position_tolerance
    cfg.rewards["goal_time_efficiency"].params["position_tolerance"] = (
      position_tolerance
    )

  if stage.get("goal_yaw_tolerance") is not None:
    yaw_tolerance = stage["goal_yaw_tolerance"]
    cfg.rewards["position_progress"].params["yaw_tolerance"] = yaw_tolerance
    cfg.rewards["yaw_progress"].params["yaw_tolerance"] = yaw_tolerance
    cfg.rewards["goal_position_bonus"].params["yaw_soft_tolerance"] = yaw_tolerance
    cfg.rewards["goal_pose_bonus"].params["yaw_tolerance"] = yaw_tolerance
    cfg.rewards["goal_time_efficiency"].params["yaw_tolerance"] = yaw_tolerance

  if stage.get("goal_hold_steps") is not None:
    hold_steps = stage["goal_hold_steps"]
    cfg.rewards["goal_pose_bonus"].params["hold_steps"] = hold_steps
    cfg.rewards["goal_time_efficiency"].params["hold_steps"] = hold_steps

  if stage.get("goal_max_lin_speed") is not None:
    max_lin_speed = stage["goal_max_lin_speed"]
    cfg.rewards["goal_pose_bonus"].params["max_lin_speed"] = max_lin_speed
    cfg.rewards["goal_time_efficiency"].params["max_lin_speed"] = max_lin_speed

  if stage.get("goal_max_yaw_rate") is not None:
    max_yaw_rate = stage["goal_max_yaw_rate"]
    cfg.rewards["goal_pose_bonus"].params["max_yaw_rate"] = max_yaw_rate
    cfg.rewards["goal_time_efficiency"].params["max_yaw_rate"] = max_yaw_rate

  if stage.get("goal_min_up_z") is not None:
    min_up_z = stage["goal_min_up_z"]
    cfg.rewards["goal_pose_bonus"].params["min_up_z"] = min_up_z
    cfg.rewards["goal_time_efficiency"].params["min_up_z"] = min_up_z

  if stage.get("near_goal_heading_distance") is not None:
    near_distance = stage["near_goal_heading_distance"]
    cfg.rewards["near_goal_heading_alignment"].params["near_distance"] = near_distance
    cfg.rewards["yaw_progress"].params["near_distance"] = near_distance

  if stage.get("near_goal_stability_weight") is not None:
    cfg.rewards["near_goal_stability"].weight = stage["near_goal_stability_weight"]
  if stage.get("near_goal_stability_distance") is not None:
    cfg.rewards["near_goal_stability"].params["near_distance"] = stage[
      "near_goal_stability_distance"
    ]
  if stage.get("near_goal_stability_lin_speed_std") is not None:
    cfg.rewards["near_goal_stability"].params["lin_speed_std"] = stage[
      "near_goal_stability_lin_speed_std"
    ]
  if stage.get("near_goal_stability_yaw_rate_std") is not None:
    cfg.rewards["near_goal_stability"].params["yaw_rate_std"] = stage[
      "near_goal_stability_yaw_rate_std"
    ]
  if stage.get("near_goal_stability_min_up_z") is not None:
    cfg.rewards["near_goal_stability"].params["min_up_z"] = stage[
      "near_goal_stability_min_up_z"
    ]

  if stage.get("goal_pose_bonus_weight") is not None:
    cfg.rewards["goal_pose_bonus"].weight = stage["goal_pose_bonus_weight"]

  if stage.get("joint_target_limits_weight") is not None:
    cfg.rewards["joint_target_limits"].weight = stage["joint_target_limits_weight"]


def _apply_t1_goal_stage_to_command_cfg(
  goal_pose_cmd: target_mdp.GoalPoseCommandCfg, stage: dict
) -> None:
  goal_pose_cmd.ranges.target_radius = stage["target_radius"]
  goal_pose_cmd.ranges.target_yaw = stage["target_yaw"]
  goal_pose_cmd.ranges.target_angle = stage["target_angle"]
  command_buckets = stage.get("target_command_buckets")
  goal_pose_cmd.ranges.target_command_buckets = (
    tuple(
      target_mdp.GoalPoseCommandCfg.CommandBucket(
        radius_range=tuple(bucket["radius_range"]),
        probability=float(bucket["probability"]),
        yaw_range=(
          tuple(bucket["yaw_range"]) if bucket["yaw_range"] is not None else None
        ),
        angle_range=(
          tuple(bucket["angle_range"]) if bucket["angle_range"] is not None else None
        ),
      )
      for bucket in command_buckets
    )
    if command_buckets is not None
    else None
  )
  radius_buckets = stage.get("target_radius_buckets")
  goal_pose_cmd.ranges.target_radius_buckets = (
    tuple(
      target_mdp.GoalPoseCommandCfg.RadiusBucket(
        radius_range=tuple(bucket["radius_range"]),
        probability=float(bucket["probability"]),
      )
      for bucket in radius_buckets
    )
    if radius_buckets is not None
    else None
  )
  goal_pose_cmd.resampling_time_range = stage["resampling_time_range"]
  goal_pose_cmd.resample_on_near_goal = stage["resample_on_near_goal"]
  goal_pose_cmd.resample_on_goal_success = stage["resample_on_goal_success"]
  goal_pose_cmd.resample_position_tolerance = stage["resample_position_tolerance"]
  goal_pose_cmd.resample_yaw_tolerance = stage["resample_yaw_tolerance"]


def _configure_t1_goal_pose_command(cfg: ManagerBasedRlEnvCfg) -> None:
  goal_pose_cmd = cfg.commands["goal_pose"]
  assert isinstance(goal_pose_cmd, target_mdp.GoalPoseCommandCfg)
  stage0 = _T1_TARGET_LOCO_GOAL_CURRICULUM_STAGES[0]
  _apply_t1_goal_stage_to_command_cfg(goal_pose_cmd, stage0)


def _configure_t1_competition_server_physics(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.sim.mujoco.integrator = "euler"
  cfg.sim.mujoco.cone = "pyramidal"
  cfg.sim.mujoco.impratio = 1.0
  cfg.sim.mujoco.iterations = 100
  cfg.sim.mujoco.ls_iterations = 50
  cfg.sim.mujoco.ccd_iterations = 35


def _t1_play_stage() -> dict:
  return _T1_TARGET_LOCO_GOAL_CURRICULUM_STAGES[-1]


def booster_t1_target_location_amp_flat_env_cfg(
  play: bool = False,
  competition_foot: bool = False,
  competition_collision: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the T1 target-location locomotion AMP task."""
  if competition_collision and competition_foot:
    raise ValueError(
      "competition_collision and competition_foot are mutually exclusive."
    )

  cfg = make_target_location_amp_env_cfg(play=play)
  robot_cfg = get_t1_robot_cfg()
  if competition_foot:
    robot_cfg = get_t1_competition_foot_robot_cfg()
  if competition_collision:
    robot_cfg = get_t1_competition_collision_robot_cfg()
  if competition_foot or competition_collision:
    _configure_t1_competition_server_physics(cfg)
  cfg.scene.entities = {
    "robot": deepcopy(robot_cfg),
  }
  cfg.scene.num_envs = 1 if play else 1024
  cfg.episode_length_s = _T1_EPISODE_LENGTH_S
  cfg.viewer.body_name = "Trunk"
  _apply_t1_joint_action_setup(
    cfg,
    action_delay_steps=(
      _T1_COMPETITION_ACTION_DELAY_STEPS if competition_collision else 0
    ),
  )
  _configure_t1_goal_rewards(cfg)
  _configure_t1_goal_pose_command(cfg)
  cfg.events["reset_robot_joints"].params["asset_cfg"] = _robot_asset_cfg()
  if not play:
    _configure_t1_domain_randomization(
      cfg,
      foot_geom_names=(
        _T1_COMPETITION_FOOT_FRICTION_GEOMS
        if competition_foot or competition_collision
        else _T1_FOOT_FRICTION_GEOMS
      ),
    )
  cfg.observations["critic"].terms["amp_robot"].params["robot_cfg"] = _robot_asset_cfg()
  cfg.terminations["fell_over"].params["height_threshold"] = _T1_FALL_HEIGHT_THRESHOLD
  cfg.terminations["fell_over"].params["up_z_threshold"] = _T1_FALL_UP_Z_THRESHOLD

  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=target_mdp.action_rate_l2,
    weight=-0.01,
  )
  cfg.rewards["joint_pos_limits"] = RewardTermCfg(
    func=target_mdp.joint_pos_limits,
    weight=-0.2,
    params={"asset_cfg": _robot_asset_cfg()},
  )
  cfg.rewards["joint_target_limits"] = RewardTermCfg(
    func=target_mdp.joint_target_pos_limits,
    weight=0.0,
    params={
      "asset_cfg": _robot_asset_cfg(),
      "action_name": "joint_pos",
    },
  )

  if not play:
    cfg.curriculum["goal_pose"] = CurriculumTermCfg(
      func=target_mdp.goal_pose_ranges,
      params={
        "command_name": "goal_pose",
        "stages": _T1_TARGET_LOCO_GOAL_CURRICULUM_STAGES,
      },
    )

  if play:
    cfg.observations["actor"].enable_corruption = False
    goal_pose_cmd = cfg.commands["goal_pose"]
    assert isinstance(goal_pose_cmd, target_mdp.GoalPoseCommandCfg)
    play_stage = _t1_play_stage()
    _apply_t1_goal_stage_to_command_cfg(goal_pose_cmd, play_stage)
    _apply_t1_goal_stage_to_reward_cfg(cfg, play_stage)
    cfg.curriculum["goal_pose"] = CurriculumTermCfg(
      func=target_mdp.goal_pose_ranges,
      params={
        "command_name": "goal_pose",
        "stages": _T1_TARGET_LOCO_GOAL_CURRICULUM_STAGES,
      },
    )

  return cfg


def booster_t1_target_location_amp_competition_foot_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the T1 target-location AMP task with competition-style foot contacts."""
  return booster_t1_target_location_amp_flat_env_cfg(
    play=play,
    competition_foot=True,
  )


def booster_t1_target_location_amp_competition_collision_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the T1 target-location AMP task with competition-style collision contacts."""
  return booster_t1_target_location_amp_flat_env_cfg(
    play=play,
    competition_collision=True,
  )
