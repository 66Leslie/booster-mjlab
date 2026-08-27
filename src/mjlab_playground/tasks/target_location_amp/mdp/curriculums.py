"""Curriculum helpers local to the target-location AMP task."""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportOptionalIterable=false

from __future__ import annotations

import math
from typing import TypedDict, cast

import torch

from .commands import GoalPoseCommandCfg


class GoalRadiusBucketStage(TypedDict):
  radius_range: tuple[float, float]
  probability: float


class GoalCommandBucketStage(TypedDict):
  radius_range: tuple[float, float]
  probability: float
  yaw_range: tuple[float, float] | None
  angle_range: tuple[float, float] | None


class GoalStage(TypedDict):
  step: int
  target_radius: tuple[float, float] | None
  target_radius_buckets: list[GoalRadiusBucketStage] | None
  target_command_buckets: list[GoalCommandBucketStage] | None
  target_yaw: tuple[float, float] | None
  target_angle: tuple[float, float] | None
  resampling_time_range: tuple[float, float] | None
  resample_on_near_goal: bool | None
  resample_on_goal_success: bool | None
  resample_position_tolerance: float | None
  resample_yaw_tolerance: float | None
  goal_position_tolerance: float | None
  goal_yaw_tolerance: float | None
  goal_hold_steps: int | None
  goal_max_lin_speed: float | None
  goal_max_yaw_rate: float | None
  goal_min_up_z: float | None
  near_goal_heading_distance: float | None
  near_goal_stability_weight: float | None
  near_goal_stability_distance: float | None
  near_goal_stability_lin_speed_std: float | None
  near_goal_stability_yaw_rate_std: float | None
  near_goal_stability_min_up_z: float | None
  goal_pose_bonus_weight: float | None
  joint_target_limits_weight: float | None


def _apply_goal_reward_stage(env, stage: GoalStage) -> None:
  reward_manager = env.reward_manager

  if stage.get("goal_position_tolerance") is not None:
    position_tolerance = float(stage["goal_position_tolerance"])
    reward_manager.get_term_cfg("position_progress").params["position_tolerance"] = (
      position_tolerance
    )
    reward_manager.get_term_cfg("yaw_progress").params["position_tolerance"] = (
      position_tolerance
    )
    reward_manager.get_term_cfg("goal_position_bonus").params["position_tolerance"] = (
      position_tolerance
    )
    reward_manager.get_term_cfg("goal_pose_bonus").params["position_tolerance"] = (
      position_tolerance
    )
    reward_manager.get_term_cfg("goal_time_efficiency").params["position_tolerance"] = (
      position_tolerance
    )

  if stage.get("goal_yaw_tolerance") is not None:
    yaw_tolerance = float(stage["goal_yaw_tolerance"])
    reward_manager.get_term_cfg("position_progress").params["yaw_tolerance"] = (
      yaw_tolerance
    )
    reward_manager.get_term_cfg("yaw_progress").params["yaw_tolerance"] = yaw_tolerance
    reward_manager.get_term_cfg("goal_position_bonus").params["yaw_soft_tolerance"] = (
      yaw_tolerance
    )
    reward_manager.get_term_cfg("goal_pose_bonus").params["yaw_tolerance"] = (
      yaw_tolerance
    )
    reward_manager.get_term_cfg("goal_time_efficiency").params["yaw_tolerance"] = (
      yaw_tolerance
    )

  if stage.get("goal_hold_steps") is not None:
    hold_steps = int(stage["goal_hold_steps"])
    reward_manager.get_term_cfg("goal_pose_bonus").params["hold_steps"] = hold_steps
    reward_manager.get_term_cfg("goal_time_efficiency").params["hold_steps"] = (
      hold_steps
    )

  if stage.get("goal_max_lin_speed") is not None:
    max_lin_speed = float(stage["goal_max_lin_speed"])
    reward_manager.get_term_cfg("goal_pose_bonus").params["max_lin_speed"] = (
      max_lin_speed
    )
    reward_manager.get_term_cfg("goal_time_efficiency").params["max_lin_speed"] = (
      max_lin_speed
    )

  if stage.get("goal_max_yaw_rate") is not None:
    max_yaw_rate = float(stage["goal_max_yaw_rate"])
    reward_manager.get_term_cfg("goal_pose_bonus").params["max_yaw_rate"] = max_yaw_rate
    reward_manager.get_term_cfg("goal_time_efficiency").params["max_yaw_rate"] = (
      max_yaw_rate
    )

  if stage.get("goal_min_up_z") is not None:
    min_up_z = float(stage["goal_min_up_z"])
    reward_manager.get_term_cfg("goal_pose_bonus").params["min_up_z"] = min_up_z
    reward_manager.get_term_cfg("goal_time_efficiency").params["min_up_z"] = min_up_z

  if stage.get("near_goal_heading_distance") is not None:
    near_distance = float(stage["near_goal_heading_distance"])
    reward_manager.get_term_cfg("near_goal_heading_alignment").params[
      "near_distance"
    ] = near_distance
    reward_manager.get_term_cfg("yaw_progress").params["near_distance"] = near_distance

  if stage.get("near_goal_stability_weight") is not None:
    reward_manager.get_term_cfg("near_goal_stability").weight = float(
      stage["near_goal_stability_weight"]
    )

  if stage.get("near_goal_stability_distance") is not None:
    reward_manager.get_term_cfg("near_goal_stability").params["near_distance"] = float(
      stage["near_goal_stability_distance"]
    )

  if stage.get("near_goal_stability_lin_speed_std") is not None:
    reward_manager.get_term_cfg("near_goal_stability").params["lin_speed_std"] = float(
      stage["near_goal_stability_lin_speed_std"]
    )

  if stage.get("near_goal_stability_yaw_rate_std") is not None:
    reward_manager.get_term_cfg("near_goal_stability").params["yaw_rate_std"] = float(
      stage["near_goal_stability_yaw_rate_std"]
    )

  if stage.get("near_goal_stability_min_up_z") is not None:
    reward_manager.get_term_cfg("near_goal_stability").params["min_up_z"] = float(
      stage["near_goal_stability_min_up_z"]
    )

  if stage.get("goal_pose_bonus_weight") is not None:
    reward_manager.get_term_cfg("goal_pose_bonus").weight = float(
      stage["goal_pose_bonus_weight"]
    )

  if stage.get("joint_target_limits_weight") is not None:
    reward_manager.get_term_cfg("joint_target_limits").weight = float(
      stage["joint_target_limits_weight"]
    )


def _radius_range_metrics(
  cfg: GoalPoseCommandCfg,
) -> tuple[float, float, torch.Tensor]:
  if cfg.ranges.target_command_buckets:
    radius_min = min(
      float(bucket.radius_range[0]) for bucket in cfg.ranges.target_command_buckets
    )
    radius_max = max(
      float(bucket.radius_range[1]) for bucket in cfg.ranges.target_command_buckets
    )
    return (
      radius_min,
      radius_max,
      torch.tensor(float(len(cfg.ranges.target_command_buckets))),
    )

  if not cfg.ranges.target_radius_buckets:
    return (
      float(cfg.ranges.target_radius[0]),
      float(cfg.ranges.target_radius[1]),
      torch.tensor(0.0),
    )

  radius_min = min(
    float(bucket.radius_range[0]) for bucket in cfg.ranges.target_radius_buckets
  )
  radius_max = max(
    float(bucket.radius_range[1]) for bucket in cfg.ranges.target_radius_buckets
  )
  return (
    radius_min,
    radius_max,
    torch.tensor(float(len(cfg.ranges.target_radius_buckets))),
  )


def goal_pose_ranges(
  env,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[GoalStage],
) -> dict[str, torch.Tensor]:
  del env_ids
  command_term = env.command_manager.get_term(command_name)
  cfg = cast(GoalPoseCommandCfg, command_term.cfg)
  reward_manager = env.reward_manager
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      if stage.get("target_radius") is not None:
        cfg.ranges.target_radius = stage["target_radius"]
      if stage.get("target_radius_buckets") is not None:
        cfg.ranges.target_radius_buckets = tuple(
          GoalPoseCommandCfg.RadiusBucket(
            radius_range=tuple(bucket["radius_range"]),
            probability=float(bucket["probability"]),
          )
          for bucket in stage["target_radius_buckets"]
        )
      if stage.get("target_command_buckets") is not None:
        cfg.ranges.target_command_buckets = tuple(
          GoalPoseCommandCfg.CommandBucket(
            radius_range=tuple(bucket["radius_range"]),
            probability=float(bucket["probability"]),
            yaw_range=(
              tuple(bucket["yaw_range"]) if bucket["yaw_range"] is not None else None
            ),
            angle_range=(
              tuple(bucket["angle_range"])
              if bucket["angle_range"] is not None
              else None
            ),
          )
          for bucket in stage["target_command_buckets"]
        )
      if stage.get("target_yaw") is not None:
        cfg.ranges.target_yaw = stage["target_yaw"]
      if stage.get("target_angle") is not None:
        cfg.ranges.target_angle = stage["target_angle"]
      if stage.get("resampling_time_range") is not None:
        cfg.resampling_time_range = stage["resampling_time_range"]
      if stage.get("resample_on_near_goal") is not None:
        cfg.resample_on_near_goal = stage["resample_on_near_goal"]
      if stage.get("resample_on_goal_success") is not None:
        cfg.resample_on_goal_success = stage["resample_on_goal_success"]
      if stage.get("resample_position_tolerance") is not None:
        cfg.resample_position_tolerance = stage["resample_position_tolerance"]
      if stage.get("resample_yaw_tolerance") is not None:
        cfg.resample_yaw_tolerance = stage["resample_yaw_tolerance"]
      _apply_goal_reward_stage(env, stage)

  radius_min, radius_max, bucket_count = _radius_range_metrics(cfg)
  return {
    "target_radius_min": torch.tensor(radius_min),
    "target_radius_max": torch.tensor(radius_max),
    "target_radius_bucket_count": bucket_count,
    "target_yaw_min": torch.tensor(cfg.ranges.target_yaw[0]),
    "target_yaw_max": torch.tensor(cfg.ranges.target_yaw[1]),
    "target_angle_min": torch.tensor(cfg.ranges.target_angle[0]),
    "target_angle_max": torch.tensor(cfg.ranges.target_angle[1]),
    "resampling_time_min": torch.tensor(cfg.resampling_time_range[0]),
    "resampling_time_max": torch.tensor(cfg.resampling_time_range[1]),
    "resample_on_near_goal": torch.tensor(float(cfg.resample_on_near_goal)),
    "resample_on_goal_success": torch.tensor(float(cfg.resample_on_goal_success)),
    "resample_position_tolerance": torch.tensor(cfg.resample_position_tolerance),
    "resample_yaw_tolerance": torch.tensor(cfg.resample_yaw_tolerance),
    "reward_goal_position_tolerance": torch.tensor(
      reward_manager.get_term_cfg("goal_pose_bonus").params["position_tolerance"]
    ),
    "reward_goal_yaw_tolerance_deg": torch.tensor(
      math.degrees(
        reward_manager.get_term_cfg("goal_pose_bonus").params["yaw_tolerance"]
      )
    ),
    "reward_goal_hold_steps": torch.tensor(
      reward_manager.get_term_cfg("goal_pose_bonus").params["hold_steps"]
    ),
  }
