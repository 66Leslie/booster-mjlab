"""Rewards local to the target-location AMP locomotion task."""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .state import update_goal_pose_state

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _log_metric(env, key: str, value: torch.Tensor) -> None:
  extras = getattr(env, "extras", None)
  if not isinstance(extras, dict):
    return
  log = extras.get("log")
  if isinstance(log, dict):
    log[key] = value


def _log_goal_pose_metrics(
  env,
  command_term,
  *,
  position_tolerance: float,
  yaw_tolerance: float,
  hold_steps: torch.Tensor,
) -> None:
  yaw_abs = torch.abs(command_term.final_heading_error)
  position_ok = command_term.distance_to_goal <= position_tolerance
  yaw_ok = yaw_abs <= yaw_tolerance
  pose_ok = position_ok & yaw_ok
  _log_metric(
    env,
    "Metrics/target_location/reward_goal_distance",
    command_term.distance_to_goal.mean(),
  )
  _log_metric(env, "Metrics/target_location/reward_final_yaw_abs", yaw_abs.mean())
  _log_metric(
    env, "Metrics/target_location/position_ok_fraction", position_ok.float().mean()
  )
  _log_metric(env, "Metrics/target_location/yaw_ok_fraction", yaw_ok.float().mean())
  _log_metric(env, "Metrics/target_location/pose_ok_fraction", pose_ok.float().mean())
  _log_metric(env, "Metrics/target_location/hold_steps_mean", hold_steps.float().mean())


def _state(env, command_name: str, position_tolerance: float, yaw_tolerance: float):
  command_term = env.command_manager.get_term(command_name)
  state = update_goal_pose_state(
    env,
    distance=command_term.distance_to_goal,
    final_yaw_error=command_term.final_heading_error,
    position_tolerance=position_tolerance,
    yaw_tolerance=yaw_tolerance,
  )
  _log_goal_pose_metrics(
    env,
    command_term,
    position_tolerance=position_tolerance,
    yaw_tolerance=yaw_tolerance,
    hold_steps=state[2],
  )
  return state


def _healthy_reward_gate(
  env,
  reward: torch.Tensor,
  *,
  entity_name: str = "robot",
  min_root_height: float | None = None,
  min_up_z: float | None = None,
) -> torch.Tensor:
  if min_root_height is None and min_up_z is None:
    return reward

  asset = env.scene[entity_name]
  stable = torch.ones_like(reward, dtype=torch.bool)
  if min_root_height is not None:
    stable = stable & (asset.data.root_link_pos_w[:, 2] >= float(min_root_height))
  if min_up_z is not None:
    up_z = -asset.data.projected_gravity_b[:, 2]
    stable = stable & (up_z >= float(min_up_z))
    _log_metric(env, "Metrics/target_location/reward_gate_up_z", up_z.mean())
  _log_metric(
    env, "Metrics/target_location/reward_gate_fraction", stable.float().mean()
  )
  # Keep negative shaping when unstable, but suppress positive reward hacking.
  return torch.where(stable, reward, torch.minimum(reward, torch.zeros_like(reward)))


def _apply_training_decay(
  env,
  reward: torch.Tensor,
  *,
  start_step: int = 0,
  decay_steps: int = 0,
  final_scale: float = 1.0,
  log_name: str | None = None,
) -> torch.Tensor:
  final = float(final_scale)
  if start_step <= 0 and decay_steps <= 0 and final >= 1.0:
    return reward

  step = int(getattr(env, "common_step_counter", 0))
  start = int(start_step)
  decay = int(decay_steps)
  if step < start:
    scale = 1.0
  elif decay <= 0:
    scale = final
  else:
    progress = min(1.0, (step - start) / float(decay))
    scale = 1.0 - (1.0 - final) * progress

  scale_tensor = torch.as_tensor(scale, dtype=reward.dtype, device=reward.device)
  if log_name is not None:
    _log_metric(env, f"Curriculum/target_location/{log_name}_scale", scale_tensor)
  return reward * scale_tensor


def alive_reward(
  env,
  decay_start_step: int = 0,
  decay_steps: int = 0,
  final_scale: float = 1.0,
) -> torch.Tensor:
  """Dense survival reward so early PPO gets a clear non-fall signal."""
  reward = torch.ones(env.num_envs, device=env.device)
  return _apply_training_decay(
    env,
    reward,
    start_step=decay_start_step,
    decay_steps=decay_steps,
    final_scale=final_scale,
    log_name="alive",
  )


def upright_orientation_reward(
  env,
  std: float = 0.5,
  decay_start_step: int = 0,
  decay_steps: int = 0,
  final_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping the root upright from projected gravity."""
  asset = env.scene[asset_cfg.name]
  gravity_xy = asset.data.projected_gravity_b[:, :2]
  error = torch.sum(torch.square(gravity_xy), dim=-1)
  reward = torch.exp(-error / max(float(std) ** 2, 1.0e-6))
  reward = _apply_training_decay(
    env,
    reward,
    start_step=decay_start_step,
    decay_steps=decay_steps,
    final_scale=final_scale,
    log_name="upright",
  )
  _log_metric(env, "Metrics/target_location/upright_reward", reward.mean())
  return reward


def root_height_reward(
  env,
  target_height: float = 0.65,
  std: float = 0.2,
  decay_start_step: int = 0,
  decay_steps: int = 0,
  final_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward staying near the nominal root height."""
  asset = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]
  error = torch.square(height - float(target_height))
  reward = torch.exp(-error / max(float(std) ** 2, 1.0e-6))
  reward = _apply_training_decay(
    env,
    reward,
    start_step=decay_start_step,
    decay_steps=decay_steps,
    final_scale=final_scale,
    log_name="root_height",
  )
  _log_metric(env, "Metrics/target_location/root_height_reward", reward.mean())
  return reward


def body_ang_vel_xy_l2(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize roll/pitch angular velocity of the root."""
  asset = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=-1)


def action_rate_l2(env) -> torch.Tensor:
  """Penalize raw policy action changes."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action),
    dim=-1,
  )


def joint_pos_limits(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excursions beyond soft joint position limits."""
  asset = env.scene[asset_cfg.name]
  soft_limits = asset.data.soft_joint_pos_limits
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  lower_limits = soft_limits[:, asset_cfg.joint_ids, 0]
  upper_limits = soft_limits[:, asset_cfg.joint_ids, 1]
  out_of_limits = -(joint_pos - lower_limits).clip(max=0.0)
  out_of_limits += (joint_pos - upper_limits).clip(min=0.0)
  return torch.sum(out_of_limits, dim=-1)


def joint_target_pos_limits(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  action_name: str = "joint_pos",
  normalize_by_range: bool = False,
) -> torch.Tensor:
  """Penalize commanded position targets beyond soft joint limits."""
  asset = env.scene[asset_cfg.name]
  term = env.action_manager.get_term(action_name)
  target_ids = term.target_ids
  soft_limits = asset.data.soft_joint_pos_limits[:, target_ids]
  lower_limits = soft_limits[:, :, 0]
  upper_limits = soft_limits[:, :, 1]

  processed_actions = term.raw_action * term.scale + term.offset
  if term.__class__.__name__ == "RelativeJointPositionAction":
    joint_pos = asset.data.joint_pos[:, target_ids]
    target_pos = joint_pos + processed_actions
  else:
    target_pos = processed_actions

  out_of_limits = -(target_pos - lower_limits).clip(max=0.0)
  out_of_limits += (target_pos - upper_limits).clip(min=0.0)
  if normalize_by_range:
    limit_range = (upper_limits - lower_limits).clamp_min(1.0e-6)
    out_of_limits = out_of_limits / limit_range
  penalty = torch.sum(out_of_limits, dim=-1)
  metric_name = (
    "Metrics/target_location/target_excess_normalized"
    if normalize_by_range
    else "Metrics/target_location/target_excess"
  )
  _log_metric(env, metric_name, penalty.mean())
  return penalty


def position_progress_reward(
  env,
  command_name: str = "goal_pose",
  position_tolerance: float = 0.1,
  yaw_tolerance: float = 0.1,
  max_speed: float = 2.5,
  entity_name: str = "robot",
  min_root_height: float | None = None,
  min_up_z: float | None = None,
) -> torch.Tensor:
  distance_progress, _, _ = _state(env, command_name, position_tolerance, yaw_tolerance)
  reward = distance_progress / max(float(env.step_dt) * max(max_speed, 1.0e-6), 1.0e-6)
  return _healthy_reward_gate(
    env,
    reward,
    entity_name=entity_name,
    min_root_height=min_root_height,
    min_up_z=min_up_z,
  )


def distance_error_penalty(
  env,
  command_name: str = "goal_pose",
  min_distance_scale: float = 0.5,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  scale = torch.clamp(
    command_term.initial_distance_to_goal,
    min=min_distance_scale,
  )
  distance_error = command_term.distance_to_goal / scale
  _log_metric(env, "Metrics/target_location/distance_error_norm", distance_error.mean())
  return distance_error


def yaw_progress_reward(
  env,
  command_name: str = "goal_pose",
  position_tolerance: float = 0.1,
  yaw_tolerance: float = 0.1,
  near_distance: float = 0.3,
  entity_name: str = "robot",
  min_root_height: float | None = None,
  min_up_z: float | None = None,
) -> torch.Tensor:
  _, yaw_progress, _ = _state(env, command_name, position_tolerance, yaw_tolerance)
  command_term = env.command_manager.get_term(command_name)
  near = (command_term.distance_to_goal <= near_distance).float()
  reward = yaw_progress * near
  return _healthy_reward_gate(
    env,
    reward,
    entity_name=entity_name,
    min_root_height=min_root_height,
    min_up_z=min_up_z,
  )


def goal_position_bonus(
  env,
  command_name: str = "goal_pose",
  position_tolerance: float = 0.1,
  yaw_soft_tolerance: float | None = None,
  entity_name: str = "robot",
  min_root_height: float | None = None,
  min_up_z: float | None = None,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  position_ok = command_term.distance_to_goal <= position_tolerance
  if yaw_soft_tolerance is None:
    reward = position_ok.float()
  else:
    yaw_ok = torch.abs(command_term.final_heading_error) <= yaw_soft_tolerance
    reward = (position_ok & yaw_ok).float()
  return _healthy_reward_gate(
    env,
    reward,
    entity_name=entity_name,
    min_root_height=min_root_height,
    min_up_z=min_up_z,
  )


def near_goal_heading_alignment_reward(
  env,
  command_name: str = "goal_pose",
  near_distance: float = 0.5,
  entity_name: str = "robot",
  min_root_height: float | None = None,
  min_up_z: float | None = None,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  near_weight = torch.exp(
    -command_term.distance_to_goal / max(float(near_distance), 1.0e-6)
  )
  reward = near_weight * torch.cos(command_term.final_heading_error)
  _log_metric(env, "Metrics/target_location/near_heading_alignment", reward.mean())
  return _healthy_reward_gate(
    env,
    reward,
    entity_name=entity_name,
    min_root_height=min_root_height,
    min_up_z=min_up_z,
  )


def goal_stability_mask(
  env,
  *,
  entity_name: str = "robot",
  max_lin_speed: float | None = None,
  max_yaw_rate: float | None = None,
  min_up_z: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  asset = env.scene[entity_name]
  device = asset.data.root_link_pos_w.device
  stable = torch.ones(
    asset.data.root_link_pos_w.shape[0], dtype=torch.bool, device=device
  )
  metrics: dict[str, torch.Tensor] = {}

  if max_lin_speed is not None:
    lin_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    metrics["lin_speed"] = lin_speed
    stable = stable & (lin_speed <= float(max_lin_speed))

  if max_yaw_rate is not None:
    yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
    metrics["yaw_rate"] = yaw_rate
    stable = stable & (yaw_rate <= float(max_yaw_rate))

  if min_up_z is not None:
    up_z = -asset.data.projected_gravity_b[:, 2]
    metrics["up_z"] = up_z
    stable = stable & (up_z >= float(min_up_z))

  metrics["stable"] = stable.float()
  return stable, metrics


def _stability_ok(
  env,
  *,
  entity_name: str,
  max_lin_speed: float | None,
  max_yaw_rate: float | None,
  min_up_z: float | None,
) -> torch.Tensor | None:
  if max_lin_speed is None and max_yaw_rate is None and min_up_z is None:
    return None

  stable, metrics = goal_stability_mask(
    env,
    entity_name=entity_name,
    max_lin_speed=max_lin_speed,
    max_yaw_rate=max_yaw_rate,
    min_up_z=min_up_z,
  )
  if "lin_speed" in metrics:
    _log_metric(
      env,
      "Metrics/target_location/stability_lin_speed",
      metrics["lin_speed"].mean(),
    )
  if "yaw_rate" in metrics:
    _log_metric(
      env,
      "Metrics/target_location/stability_yaw_rate",
      metrics["yaw_rate"].mean(),
    )
  if "up_z" in metrics:
    _log_metric(env, "Metrics/target_location/stability_up_z", metrics["up_z"].mean())
  _log_metric(
    env, "Metrics/target_location/stability_ok_fraction", stable.float().mean()
  )
  return stable


def goal_pose_terminal_bonus(
  env,
  command_name: str = "goal_pose",
  position_tolerance: float = 0.1,
  yaw_tolerance: float = 0.1,
  hold_steps: int = 10,
  entity_name: str = "robot",
  max_lin_speed: float | None = None,
  max_yaw_rate: float | None = None,
  min_up_z: float | None = None,
) -> torch.Tensor:
  from .state import update_goal_pose_success_hold_state

  command_term = env.command_manager.get_term(command_name)
  stable = _stability_ok(
    env,
    entity_name=entity_name,
    max_lin_speed=max_lin_speed,
    max_yaw_rate=max_yaw_rate,
    min_up_z=min_up_z,
  )
  success_now = (command_term.distance_to_goal <= float(position_tolerance)) & (
    torch.abs(command_term.final_heading_error) <= float(yaw_tolerance)
  )
  if stable is not None:
    success_now = success_now & stable
  hold = update_goal_pose_success_hold_state(env, success_now=success_now)
  just_reached = hold == hold_steps
  _log_metric(
    env,
    "Metrics/target_location/terminal_success_fraction",
    just_reached.float().mean(),
  )
  return just_reached.float()


def goal_time_efficiency_bonus(
  env,
  command_name: str = "goal_pose",
  position_tolerance: float = 0.1,
  yaw_tolerance: float = 0.1,
  hold_steps: int = 10,
  max_speed: float = 2.5,
  entity_name: str = "robot",
  max_lin_speed: float | None = None,
  max_yaw_rate: float | None = None,
  min_up_z: float | None = None,
) -> torch.Tensor:
  from .state import update_goal_pose_success_hold_state

  command_term = env.command_manager.get_term(command_name)
  stable = _stability_ok(
    env,
    entity_name=entity_name,
    max_lin_speed=max_lin_speed,
    max_yaw_rate=max_yaw_rate,
    min_up_z=min_up_z,
  )
  success_now = (command_term.distance_to_goal <= float(position_tolerance)) & (
    torch.abs(command_term.final_heading_error) <= float(yaw_tolerance)
  )
  if stable is not None:
    success_now = success_now & stable
  hold = update_goal_pose_success_hold_state(env, success_now=success_now)
  just_reached = hold == hold_steps

  optimal_time = command_term.initial_distance_to_goal / max(max_speed, 1.0e-6)
  actual_steps = env.episode_length_buf.float() - float(max(hold_steps - 1, 0))
  actual_steps = torch.clamp(actual_steps, min=1.0)
  actual_time = actual_steps * float(env.step_dt)

  efficiency = torch.clamp(optimal_time / actual_time, min=0.0, max=1.0)
  bonus = torch.where(just_reached, efficiency, torch.zeros_like(efficiency))
  _log_metric(
    env,
    "Metrics/target_location/time_efficiency_on_success",
    torch.where(just_reached, efficiency, torch.zeros_like(efficiency)).mean(),
  )
  return bonus


def near_goal_stability_reward(
  env,
  command_name: str = "goal_pose",
  entity_name: str = "robot",
  near_distance: float = 0.5,
  position_tolerance: float | None = None,
  yaw_tolerance: float | None = None,
  lin_speed_std: float = 0.35,
  yaw_rate_std: float = 0.6,
  min_up_z: float = 0.75,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  asset = env.scene[entity_name]
  lin_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
  yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
  up_z = -asset.data.projected_gravity_b[:, 2]

  near_weight = torch.exp(
    -command_term.distance_to_goal / max(float(near_distance), 1.0e-6)
  )
  lin_score = torch.exp(-torch.square(lin_speed / max(float(lin_speed_std), 1.0e-6)))
  yaw_score = torch.exp(-torch.square(yaw_rate / max(float(yaw_rate_std), 1.0e-6)))
  upright_score = torch.clamp(
    (up_z - float(min_up_z)) / max(1.0 - float(min_up_z), 1.0e-6),
    min=0.0,
    max=1.0,
  )
  reward = near_weight * lin_score * yaw_score * upright_score
  if position_tolerance is not None:
    reward = (
      reward * (command_term.distance_to_goal <= float(position_tolerance)).float()
    )
  if yaw_tolerance is not None:
    reward = (
      reward
      * (torch.abs(command_term.final_heading_error) <= float(yaw_tolerance)).float()
    )
  _log_metric(env, "Metrics/target_location/near_stability_reward", reward.mean())
  return reward
