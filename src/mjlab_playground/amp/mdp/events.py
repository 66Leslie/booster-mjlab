"""Reset events shared by locomotion AMP tasks."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz


def _env_ids(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
  all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
  if env_ids is None:
    return all_ids
  if isinstance(env_ids, slice):
    return all_ids[env_ids]
  return env_ids


def reset_robot_near_origin(
  env,
  env_ids: torch.Tensor | slice | None,
  robot_name: str = "robot",
  x_range: tuple[float, float] = (-0.03, 0.03),
  y_range: tuple[float, float] = (-0.03, 0.03),
  yaw_range: tuple[float, float] = (-0.15, 0.15),
  z_offset: float = 0.02,
  z_offset_range: tuple[float, float] | None = None,
  roll_range: tuple[float, float] = (0.0, 0.0),
  pitch_range: tuple[float, float] = (0.0, 0.0),
  root_lin_vel_xy_range: tuple[float, float] = (0.0, 0.0),
  root_lin_vel_z_range: tuple[float, float] = (0.0, 0.0),
  root_ang_vel_range: tuple[float, float] = (0.0, 0.0),
  phase_period_s: float | None = None,
) -> None:
  """Reset a standing robot near its environment origin with optional noise."""
  del phase_period_s
  resolved_ids = _env_ids(env, env_ids)
  robot = env.scene[robot_name]
  num_resets = len(resolved_ids)

  default_root_state = robot.data.default_root_state[resolved_ids].clone()
  positions = env.scene.env_origins[resolved_ids] + default_root_state[:, :3]
  positions[:, 0] += torch.empty(num_resets, device=env.device).uniform_(*x_range)
  positions[:, 1] += torch.empty(num_resets, device=env.device).uniform_(*y_range)
  if z_offset_range is None:
    positions[:, 2] += float(z_offset)
  else:
    positions[:, 2] += torch.empty(num_resets, device=env.device).uniform_(
      *z_offset_range
    )

  roll = torch.empty(num_resets, device=env.device).uniform_(*roll_range)
  pitch = torch.empty(num_resets, device=env.device).uniform_(*pitch_range)
  yaw = torch.empty(num_resets, device=env.device).uniform_(*yaw_range)
  quat = quat_from_euler_xyz(roll, pitch, yaw)
  robot.write_root_link_pose_to_sim(
    torch.cat((positions, quat), dim=-1), env_ids=resolved_ids
  )

  root_velocity = torch.empty(num_resets, 6, device=env.device)
  root_velocity[:, :2].uniform_(*root_lin_vel_xy_range)
  root_velocity[:, 2].uniform_(*root_lin_vel_z_range)
  root_velocity[:, 3:].uniform_(*root_ang_vel_range)
  robot.write_root_link_velocity_to_sim(root_velocity, env_ids=resolved_ids)
