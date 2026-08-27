"""Termination helpers shared by locomotion AMP tasks."""

from __future__ import annotations

import torch


def fell_over(
  env,
  robot_name: str = "robot",
  height_threshold: float = 0.38,
  up_z_threshold: float = 0.55,
) -> torch.Tensor:
  """Return environments whose robot is too low or too tilted."""
  robot = env.scene[robot_name]
  height = robot.data.root_link_pos_w[:, 2]
  up_z = -robot.data.projected_gravity_b[:, 2]
  return (height < height_threshold) | (up_z < up_z_threshold)
