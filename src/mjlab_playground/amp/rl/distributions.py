"""Action distributions used by AMP/SMP robot-control tasks."""

from __future__ import annotations

import math

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal


class ClampedGaussianDistribution(GaussianDistribution):
  """Gaussian distribution with a bounded state-independent standard deviation."""

  def __init__(
    self,
    output_dim: int,
    init_std: float = 0.45,
    std_type: str = "log",
    min_std: float = 0.05,
    max_std: float = 1.0,
  ) -> None:
    if min_std <= 0.0:
      raise ValueError(f"min_std must be positive, got {min_std}.")
    if max_std < min_std:
      raise ValueError(
        f"max_std must be greater than or equal to min_std, got {max_std} < {min_std}."
      )
    super().__init__(output_dim=output_dim, init_std=init_std, std_type=std_type)
    self.min_std = float(min_std)
    self.max_std = float(max_std)

  def update(self, mlp_output: torch.Tensor) -> None:
    """Update the Gaussian distribution from MLP output."""
    mean = mlp_output
    if self.std_type == "scalar":
      std = torch.clamp(self.std_param, self.min_std, self.max_std).expand_as(mean)
    elif self.std_type == "log":
      log_std = torch.clamp(
        self.log_std_param,
        math.log(self.min_std),
        math.log(self.max_std),
      )
      std = torch.exp(log_std).expand_as(mean)
    else:
      raise ValueError(
        f"Unknown standard deviation type: {self.std_type}. Should be 'scalar' or 'log'."
      )
    self._distribution = Normal(mean, std)
