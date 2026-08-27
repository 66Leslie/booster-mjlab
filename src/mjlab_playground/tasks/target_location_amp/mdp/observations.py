"""Goal-pose observations for target-location AMP locomotion."""

from __future__ import annotations

import torch


def goal_local_position(env, command_name: str = "goal_pose") -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  return command_term.local_pos_error


def heading_to_target_features(env, command_name: str = "goal_pose") -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  angle = command_term.heading_to_target
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def final_yaw_error_features(env, command_name: str = "goal_pose") -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  angle = command_term.final_heading_error
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def goal_distance(env, command_name: str = "goal_pose") -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  return command_term.distance_to_goal.unsqueeze(-1)
