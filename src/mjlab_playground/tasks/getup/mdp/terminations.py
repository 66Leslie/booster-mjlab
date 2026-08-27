"""Termination conditions for the getup task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def energy_termination(
  env: ManagerBasedRlEnv,
  threshold: float = float("inf"),
  settle_steps: int = 0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when mechanical power exceeds threshold.

  Power = sum(|actuator_force * joint_vel|). Skips the first settle_steps steps so
  drop/settle dynamics don't trigger early termination.
  """
  asset: Entity = env.scene[asset_cfg.name]
  power = torch.sum(
    torch.abs(
      asset.data.actuator_force[:, asset_cfg.actuator_ids]
      * asset.data.joint_vel[:, asset_cfg.joint_ids]
    ),
    dim=-1,
  )
  past_settle = env.episode_length_buf > settle_steps
  return past_settle & (power > threshold)
