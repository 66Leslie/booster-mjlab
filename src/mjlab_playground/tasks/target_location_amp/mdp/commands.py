"""Goal-pose command local to the target-location AMP task."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from .rewards import goal_stability_mask
from .state import reset_goal_pose_state, update_goal_pose_success_hold_state


class GoalPoseCommand(CommandTerm):
  cfg: "GoalPoseCommandCfg"

  def __init__(self, cfg: "GoalPoseCommandCfg", env):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.target_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.target_yaw_w = torch.zeros(self.num_envs, device=self.device)
    self.initial_distance_to_goal = torch.zeros(self.num_envs, device=self.device)
    self._target_offset_b = torch.zeros(self.num_envs, 2, device=self.device)
    self._target_yaw_offset = torch.zeros(self.num_envs, device=self.device)
    self._target_needs_anchor = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.local_pos_error = torch.zeros(self.num_envs, 2, device=self.device)
    self.distance_to_goal = torch.zeros(self.num_envs, device=self.device)
    self.heading_to_target = torch.zeros(self.num_envs, device=self.device)
    self.final_heading_error = torch.zeros(self.num_envs, device=self.device)
    self._resample_on_next_step = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goal_final_yaw_abs"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      (
        self.local_pos_error,
        self.final_heading_error.unsqueeze(-1),
      ),
      dim=-1,
    )

  def _update_metrics(self) -> None:
    self.metrics["goal_distance"] = self.distance_to_goal
    self.metrics["goal_final_yaw_abs"] = torch.abs(self.final_heading_error)

  def _sample_bucket_ids(
    self,
    probabilities: list[float],
    num_samples: int,
    error_message: str,
  ) -> torch.Tensor:
    probs = torch.tensor(
      [max(float(probability), 0.0) for probability in probabilities],
      dtype=torch.float32,
      device=self.device,
    )
    total_probability = float(probs.sum().item())
    if total_probability <= 0.0:
      raise ValueError(error_message)
    probs = probs / total_probability
    return torch.multinomial(probs, num_samples, replacement=True)

  def _sample_goal_components(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    command_buckets = self.cfg.ranges.target_command_buckets
    if command_buckets:
      bucket_ids = self._sample_bucket_ids(
        [bucket.probability for bucket in command_buckets],
        len(env_ids),
        "target_command_buckets must contain positive probability mass.",
      )
      radius_min = torch.tensor(
        [float(bucket.radius_range[0]) for bucket in command_buckets],
        dtype=torch.float32,
        device=self.device,
      )[bucket_ids]
      radius_max = torch.tensor(
        [float(bucket.radius_range[1]) for bucket in command_buckets],
        dtype=torch.float32,
        device=self.device,
      )[bucket_ids]
      angle_min = torch.tensor(
        [
          float(
            bucket.angle_range[0]
            if bucket.angle_range is not None
            else self.cfg.ranges.target_angle[0]
          )
          for bucket in command_buckets
        ],
        dtype=torch.float32,
        device=self.device,
      )[bucket_ids]
      angle_max = torch.tensor(
        [
          float(
            bucket.angle_range[1]
            if bucket.angle_range is not None
            else self.cfg.ranges.target_angle[1]
          )
          for bucket in command_buckets
        ],
        dtype=torch.float32,
        device=self.device,
      )[bucket_ids]
      yaw_min = torch.tensor(
        [
          float(
            bucket.yaw_range[0]
            if bucket.yaw_range is not None
            else self.cfg.ranges.target_yaw[0]
          )
          for bucket in command_buckets
        ],
        dtype=torch.float32,
        device=self.device,
      )[bucket_ids]
      yaw_max = torch.tensor(
        [
          float(
            bucket.yaw_range[1]
            if bucket.yaw_range is not None
            else self.cfg.ranges.target_yaw[1]
          )
          for bucket in command_buckets
        ],
        dtype=torch.float32,
        device=self.device,
      )[bucket_ids]
      target_radius = radius_min + torch.rand(len(env_ids), device=self.device) * (
        radius_max - radius_min
      )
      target_angle = angle_min + torch.rand(len(env_ids), device=self.device) * (
        angle_max - angle_min
      )
      yaw_offset = yaw_min + torch.rand(len(env_ids), device=self.device) * (
        yaw_max - yaw_min
      )
      return target_radius, target_angle, yaw_offset

    target_radius = self._sample_target_radius(env_ids)
    target_angle = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.ranges.target_angle
    )
    yaw_offset = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.ranges.target_yaw
    )
    return target_radius, target_angle, yaw_offset

  def _sample_target_radius(self, env_ids: torch.Tensor) -> torch.Tensor:
    buckets = self.cfg.ranges.target_radius_buckets
    if not buckets:
      return torch.empty(len(env_ids), device=self.device).uniform_(
        *self.cfg.ranges.target_radius
      )

    bucket_ids = self._sample_bucket_ids(
      [bucket.probability for bucket in buckets],
      len(env_ids),
      "target_radius_buckets must contain positive probability mass.",
    )
    radius_min = torch.tensor(
      [float(bucket.radius_range[0]) for bucket in buckets],
      dtype=torch.float32,
      device=self.device,
    )[bucket_ids]
    radius_max = torch.tensor(
      [float(bucket.radius_range[1]) for bucket in buckets],
      dtype=torch.float32,
      device=self.device,
    )[bucket_ids]
    return radius_min + torch.rand(len(env_ids), device=self.device) * (
      radius_max - radius_min
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return

    target_radius, target_angle, yaw_offset = self._sample_goal_components(env_ids)
    target_x = target_radius * torch.cos(target_angle)
    target_y = target_radius * torch.sin(target_angle)
    self.initial_distance_to_goal[env_ids] = target_radius
    self._target_offset_b[env_ids, 0] = target_x
    self._target_offset_b[env_ids, 1] = target_y
    self._target_yaw_offset[env_ids] = yaw_offset
    self._target_needs_anchor[env_ids] = True
    self._resample_on_next_step[env_ids] = False
    reset_goal_pose_state(self._env, env_ids)

  def _anchor_pending_targets(self) -> None:
    heading = self.robot.data.heading_w
    root_pos_w = self.robot.data.root_link_pos_w[:, :2]
    if torch.any(self._target_needs_anchor):
      env_ids = self._target_needs_anchor.nonzero(as_tuple=False).flatten()
      cos_h = torch.cos(heading[env_ids])
      sin_h = torch.sin(heading[env_ids])
      target_offset_b = self._target_offset_b[env_ids]
      delta_w = torch.stack(
        (
          cos_h * target_offset_b[:, 0] - sin_h * target_offset_b[:, 1],
          sin_h * target_offset_b[:, 0] + cos_h * target_offset_b[:, 1],
        ),
        dim=-1,
      )
      self.target_pos_w[env_ids] = root_pos_w[env_ids] + delta_w
      self.target_yaw_w[env_ids] = wrap_to_pi(
        heading[env_ids] + self._target_yaw_offset[env_ids]
      )
      self._target_needs_anchor[env_ids] = False

  def _refresh_local_errors(self) -> None:
    root_pos_w = self.robot.data.root_link_pos_w[:, :2]
    heading = self.robot.data.heading_w
    delta_w = self.target_pos_w - root_pos_w
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    self.local_pos_error[:, 0] = cos_h * delta_w[:, 0] + sin_h * delta_w[:, 1]
    self.local_pos_error[:, 1] = -sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
    self.distance_to_goal = torch.norm(self.local_pos_error, dim=-1)
    self.heading_to_target = torch.atan2(
      self.local_pos_error[:, 1], self.local_pos_error[:, 0]
    )
    self.final_heading_error = wrap_to_pi(self.target_yaw_w - heading)

  def _queue_near_goal_resample(self) -> None:
    if not self.cfg.resample_on_near_goal:
      return
    reached_position = self.distance_to_goal <= self.cfg.resample_position_tolerance
    reached_yaw = torch.abs(self.final_heading_error) <= self.cfg.resample_yaw_tolerance
    self._resample_on_next_step |= reached_position & reached_yaw

  def _queue_goal_success_resample(self) -> None:
    if not self.cfg.resample_on_goal_success:
      return

    reward_manager = getattr(self._env, "reward_manager", None)
    if reward_manager is None:
      return
    try:
      goal_pose_bonus_cfg = reward_manager.get_term_cfg("goal_pose_bonus")
    except (KeyError, ValueError, AttributeError):
      return
    params = getattr(goal_pose_bonus_cfg, "params", None)
    if not isinstance(params, dict):
      return

    position_tolerance = float(
      params.get("position_tolerance", self.cfg.resample_position_tolerance)
    )
    yaw_tolerance = float(params.get("yaw_tolerance", self.cfg.resample_yaw_tolerance))
    hold_steps = int(params.get("hold_steps", 1))
    entity_name = str(params.get("entity_name", self.cfg.entity_name))
    max_lin_speed = params.get("max_lin_speed")
    max_yaw_rate = params.get("max_yaw_rate")
    min_up_z = params.get("min_up_z")

    reached_now = (self.distance_to_goal <= position_tolerance) & (
      torch.abs(self.final_heading_error) <= yaw_tolerance
    )
    if max_lin_speed is not None or max_yaw_rate is not None or min_up_z is not None:
      stable, _ = goal_stability_mask(
        self._env,
        entity_name=entity_name,
        max_lin_speed=(float(max_lin_speed) if max_lin_speed is not None else None),
        max_yaw_rate=(float(max_yaw_rate) if max_yaw_rate is not None else None),
        min_up_z=float(min_up_z) if min_up_z is not None else None,
      )
      reached_now = reached_now & stable
    hold = update_goal_pose_success_hold_state(
      self._env,
      success_now=reached_now,
    )
    reached = hold >= hold_steps
    self._resample_on_next_step |= reached

  def _update_command(self) -> None:
    if torch.any(self._resample_on_next_step):
      env_ids = self._resample_on_next_step.nonzero(as_tuple=False).flatten()
      self._resample(env_ids)
    self._anchor_pending_targets()
    self._refresh_local_errors()
    self._queue_near_goal_resample()
    self._queue_goal_success_resample()

  def _debug_vis_impl(self, visualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    root_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    target_pos_ws = self.target_pos_w.cpu().numpy()
    target_yaws = self.target_yaw_w.cpu().numpy()

    z_offset = self.cfg.viz.z_offset
    sphere_radius = self.cfg.viz.sphere_radius
    heading_scale = self.cfg.viz.heading_scale
    link_radius = self.cfg.viz.link_radius

    for env_idx in env_indices:
      root_pos_w = root_pos_ws[env_idx]
      target_pos_w = target_pos_ws[env_idx]

      if np.linalg.norm(root_pos_w) < 1.0e-6:
        continue

      target_point = np.array(
        [target_pos_w[0], target_pos_w[1], z_offset], dtype=np.float32
      )
      robot_point = np.array([root_pos_w[0], root_pos_w[1], z_offset], dtype=np.float32)
      target_yaw = float(target_yaws[env_idx])

      visualizer.add_sphere(
        target_point,
        radius=sphere_radius,
        color=(1.0, 0.55, 0.1, 0.9),
        label=f"goal_target_{env_idx}",
      )
      visualizer.add_cylinder(
        robot_point,
        target_point,
        radius=link_radius,
        color=(1.0, 0.7, 0.2, 0.35),
        label=f"goal_link_{env_idx}",
      )

      heading_end = np.array(
        [
          target_pos_w[0] + heading_scale * np.cos(target_yaw),
          target_pos_w[1] + heading_scale * np.sin(target_yaw),
          z_offset,
        ],
        dtype=np.float32,
      )
      visualizer.add_arrow(
        target_point,
        heading_end,
        color=(1.0, 0.2, 0.2, 0.95),
        width=link_radius * 1.5,
        label=f"goal_heading_{env_idx}",
      )


@dataclass(kw_only=True)
class GoalPoseCommandCfg(CommandTermCfg):
  @dataclass
  class RadiusBucket:
    radius_range: tuple[float, float]
    probability: float

  @dataclass
  class CommandBucket:
    radius_range: tuple[float, float]
    probability: float
    yaw_range: tuple[float, float] | None = None
    angle_range: tuple[float, float] | None = None

  entity_name: str
  resample_on_near_goal: bool = False
  resample_on_goal_success: bool = False
  resample_position_tolerance: float = 0.0
  resample_yaw_tolerance: float = 0.0

  @dataclass
  class Ranges:
    target_radius: tuple[float, float]
    target_yaw: tuple[float, float]
    target_angle: tuple[float, float] = (-math.pi, math.pi)
    target_radius_buckets: tuple["GoalPoseCommandCfg.RadiusBucket", ...] | None = None
    target_command_buckets: tuple["GoalPoseCommandCfg.CommandBucket", ...] | None = None

  @dataclass
  class VizCfg:
    z_offset: float = 0.08
    sphere_radius: float = 0.07
    heading_scale: float = 0.28
    link_radius: float = 0.012

  ranges: Ranges
  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env) -> GoalPoseCommand:
    return GoalPoseCommand(self, env)
