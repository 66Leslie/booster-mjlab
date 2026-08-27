"""Terminations local to the target-location AMP locomotion task."""

from __future__ import annotations

from .rewards import goal_stability_mask
from .state import update_goal_pose_success_hold_state


def goal_pose_reached(
  env,
  command_name: str = "goal_pose",
  position_tolerance: float = 0.1,
  yaw_tolerance: float = 0.2,
  hold_steps: int = 10,
  entity_name: str = "robot",
  max_lin_speed: float | None = None,
  max_yaw_rate: float | None = None,
  min_up_z: float | None = None,
):
  command_term = env.command_manager.get_term(command_name)
  success_now = (command_term.distance_to_goal <= float(position_tolerance)) & (
    abs(command_term.final_heading_error) <= float(yaw_tolerance)
  )
  if max_lin_speed is not None or max_yaw_rate is not None or min_up_z is not None:
    stable, _ = goal_stability_mask(
      env,
      entity_name=entity_name,
      max_lin_speed=max_lin_speed,
      max_yaw_rate=max_yaw_rate,
      min_up_z=min_up_z,
    )
    success_now = success_now & stable
  hold = update_goal_pose_success_hold_state(env, success_now=success_now)
  return hold >= hold_steps
