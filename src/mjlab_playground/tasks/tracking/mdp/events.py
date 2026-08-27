"""Reset event helpers for tracking tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


TRACKING_HOME_RESET_MASK_KEY = "tracking_home_reset_mask"


def sample_tracking_reset_source(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  home_probability: float,
  mask_key: str = TRACKING_HOME_RESET_MASK_KEY,
) -> None:
  """Sample which envs should use the home-pose reset path.

  This event does not write simulation state directly. It records a boolean mask in
  ``env.extras``. The tracking motion command consumes this mask during command
  resampling and applies either:
  - the default motion-frame reset path, or
  - the home-pose reset path.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)

  p = float(max(0.0, min(1.0, home_probability)))

  mask = env.extras.get(mask_key)
  if not isinstance(mask, torch.Tensor) or mask.shape != (env.num_envs,):
    mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.extras[mask_key] = mask

  if p <= 0.0:
    mask[env_ids] = False
    return
  if p >= 1.0:
    mask[env_ids] = True
    return

  mask[env_ids] = torch.rand(len(env_ids), device=env.device) < p
