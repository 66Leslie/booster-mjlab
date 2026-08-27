# Adapted from KaydenKnapik/BoosterT1mjlab (Apache-2.0); modified for this fork.
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Expose command axes plus heading error encoding.

  The velocity task mixes three mutually exclusive command modes: stand,
  x+heading, and lateral-y without heading. Both linear command axes are
  exposed directly and heading stays in sin/cos form for wraparound safety.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None
  command_term = env.command_manager.get_term(command_name)
  heading_target = getattr(command_term, "heading_target", None)
  assert heading_target is not None, (
    f"Command term '{command_name}' does not expose a heading target. "
    "Enable heading_command for this task."
  )

  robot = getattr(command_term, "robot", None)
  if robot is None:
    robot = env.scene["robot"]
  heading_error = wrap_to_pi(heading_target - robot.data.heading_w)
  is_heading_env = getattr(command_term, "is_heading_env", None)
  if is_heading_env is not None:
    heading_error = torch.where(
      is_heading_env,
      heading_error,
      torch.zeros_like(heading_error),
    )
  is_standing_env = getattr(command_term, "is_standing_env", None)
  if is_standing_env is not None:
    heading_error = torch.where(
      is_standing_env,
      torch.zeros_like(heading_error),
      heading_error,
    )

  return torch.stack(
    (command[:, 0], command[:, 1], torch.sin(heading_error), torch.cos(heading_error)),
    dim=1,
  )


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
