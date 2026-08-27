"""Reward and metric helpers shared by locomotion AMP tasks."""

# pyright: reportOptionalMemberAccess=false

from __future__ import annotations

import torch

from .terminations import fell_over as termination_fell_over


def fell_over_penalty(
  env,
  robot_name: str = "robot",
  height_threshold: float = 0.38,
  up_z_threshold: float = 0.55,
) -> torch.Tensor:
  """Return a fall indicator aligned with the current-step termination."""
  termination_manager = getattr(env, "termination_manager", None)
  try:
    fell = termination_manager.get_term("fell_over")
  except (AttributeError, KeyError):
    fell = termination_fell_over(
      env,
      robot_name=robot_name,
      height_threshold=height_threshold,
      up_z_threshold=up_z_threshold,
    )
  env.extras.setdefault("log", {})["Metrics/robot/fall_fraction"] = fell.float().mean()
  if bool(getattr(env.cfg, "scale_rewards_by_dt", True)):
    return fell.float() / max(float(env.step_dt), 1.0e-6)
  return fell.float()


def fell_over_fraction(
  env,
  robot_name: str = "robot",
  height_threshold: float = 0.38,
  up_z_threshold: float = 0.55,
) -> torch.Tensor:
  """Per-step fall indicator for episode metrics."""
  return termination_fell_over(
    env,
    robot_name=robot_name,
    height_threshold=height_threshold,
    up_z_threshold=up_z_threshold,
  ).float()
