"""Goal-pose progress state for target-location AMP locomotion."""

from __future__ import annotations

import torch


def ensure_goal_pose_state(env) -> None:
  if not hasattr(env, "_goal_pose_prev_distance"):
    env._goal_pose_prev_distance = torch.zeros(
      env.num_envs, dtype=torch.float32, device=env.device
    )
  if not hasattr(env, "_goal_pose_prev_yaw_abs"):
    env._goal_pose_prev_yaw_abs = torch.zeros(
      env.num_envs, dtype=torch.float32, device=env.device
    )
  if not hasattr(env, "_goal_pose_distance_progress"):
    env._goal_pose_distance_progress = torch.zeros(
      env.num_envs, dtype=torch.float32, device=env.device
    )
  if not hasattr(env, "_goal_pose_yaw_progress"):
    env._goal_pose_yaw_progress = torch.zeros(
      env.num_envs, dtype=torch.float32, device=env.device
    )
  if not hasattr(env, "_goal_pose_hold_steps"):
    env._goal_pose_hold_steps = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
  if not hasattr(env, "_goal_pose_success_hold_steps"):
    env._goal_pose_success_hold_steps = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
  if not hasattr(env, "_goal_pose_success_hold_last_step"):
    env._goal_pose_success_hold_last_step = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
  if not hasattr(env, "_goal_pose_state_last_step"):
    env._goal_pose_state_last_step = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
  if not hasattr(env, "_goal_pose_just_resampled"):
    env._goal_pose_just_resampled = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )


def reset_goal_pose_state(env, env_ids: torch.Tensor) -> None:
  ensure_goal_pose_state(env)
  env._goal_pose_prev_distance[env_ids] = 0.0
  env._goal_pose_prev_yaw_abs[env_ids] = 0.0
  env._goal_pose_distance_progress[env_ids] = 0.0
  env._goal_pose_yaw_progress[env_ids] = 0.0
  env._goal_pose_hold_steps[env_ids] = 0
  env._goal_pose_success_hold_steps[env_ids] = 0
  env._goal_pose_success_hold_last_step[env_ids] = -1
  env._goal_pose_state_last_step[env_ids] = -1
  env._goal_pose_just_resampled[env_ids] = True


def update_goal_pose_state(
  env,
  *,
  distance: torch.Tensor,
  final_yaw_error: torch.Tensor,
  position_tolerance: float,
  yaw_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  ensure_goal_pose_state(env)

  current_step = env.episode_length_buf
  step_changed = current_step != env._goal_pose_state_last_step
  if torch.any(step_changed):
    yaw_abs = torch.abs(final_yaw_error)
    skip_mask = (current_step <= 1) | env._goal_pose_just_resampled

    distance_progress = env._goal_pose_prev_distance - distance
    yaw_progress = env._goal_pose_prev_yaw_abs - yaw_abs
    distance_progress = torch.where(
      skip_mask,
      torch.zeros_like(distance_progress),
      distance_progress,
    )
    yaw_progress = torch.where(
      skip_mask,
      torch.zeros_like(yaw_progress),
      yaw_progress,
    )

    position_ok = distance <= position_tolerance
    yaw_ok = yaw_abs <= yaw_tolerance
    goal_ok = position_ok & yaw_ok
    next_hold_steps = torch.where(
      goal_ok,
      env._goal_pose_hold_steps + 1,
      torch.zeros_like(env._goal_pose_hold_steps),
    )

    env._goal_pose_distance_progress = torch.where(
      step_changed, distance_progress, env._goal_pose_distance_progress
    )
    env._goal_pose_yaw_progress = torch.where(
      step_changed, yaw_progress, env._goal_pose_yaw_progress
    )
    env._goal_pose_hold_steps = torch.where(
      step_changed, next_hold_steps, env._goal_pose_hold_steps
    )
    env._goal_pose_prev_distance = torch.where(
      step_changed, distance, env._goal_pose_prev_distance
    )
    env._goal_pose_prev_yaw_abs = torch.where(
      step_changed, yaw_abs, env._goal_pose_prev_yaw_abs
    )
    env._goal_pose_state_last_step = torch.where(
      step_changed, current_step, env._goal_pose_state_last_step
    )
    env._goal_pose_just_resampled[step_changed] = False

  return (
    env._goal_pose_distance_progress,
    env._goal_pose_yaw_progress,
    env._goal_pose_hold_steps,
  )


def update_goal_pose_success_hold_state(
  env,
  *,
  success_now: torch.Tensor,
) -> torch.Tensor:
  """Count consecutive controller steps satisfying the full success contract."""
  ensure_goal_pose_state(env)
  current_step = env.episode_length_buf
  step_changed = current_step != env._goal_pose_success_hold_last_step
  next_hold_steps = torch.where(
    success_now,
    env._goal_pose_success_hold_steps + 1,
    torch.zeros_like(env._goal_pose_success_hold_steps),
  )
  env._goal_pose_success_hold_steps = torch.where(
    step_changed,
    next_hold_steps,
    env._goal_pose_success_hold_steps,
  )
  env._goal_pose_success_hold_last_step = torch.where(
    step_changed,
    current_step,
    env._goal_pose_success_hold_last_step,
  )
  return env._goal_pose_success_hold_steps
