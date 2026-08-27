"""Discriminator network for adversarial motion priors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import MLP, EmpiricalNormalization


class AmpDiscriminator(nn.Module):
  """Small AMP discriminator using consecutive robot-only states."""

  def __init__(
    self,
    amp_obs_dim: int,
    hidden_dims: tuple[int, ...] = (512, 256),
    activation: str = "relu",
    amp_reward_coef: float = 0.3,
    amp_reward_scale: float = 1.0,
    task_reward_lerp: float = 0.7,
    grad_penalty_coef: float = 10.0,
    normalize_input: bool = True,
    loss_type: str = "bce",
    reward_type: str = "softplus",
    expert_label: float = 0.9,
    policy_label: float = 0.1,
    reward_clip_max: float = 1.5,
  ) -> None:
    super().__init__()
    self.amp_obs_dim = int(amp_obs_dim)
    self.input_dim = 2 * self.amp_obs_dim
    self.amp_reward_coef = float(amp_reward_coef)
    self.amp_reward_scale = float(amp_reward_scale)
    self.task_reward_lerp = float(task_reward_lerp)
    self.grad_penalty_coef = float(grad_penalty_coef)
    self.loss_type = str(loss_type)
    self.reward_type = str(reward_type)
    self.expert_label = float(expert_label)
    self.policy_label = float(policy_label)
    self.reward_clip_max = float(reward_clip_max)
    self.normalizer = (
      EmpiricalNormalization(self.input_dim) if normalize_input else nn.Identity()
    )
    self.net = MLP(self.input_dim, 1, hidden_dims, activation)

  def forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    x = torch.cat((state, next_state), dim=-1)
    return self.net(self.normalizer(x))

  @torch.no_grad()
  def predict_style_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    amp_reward_scale: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the AMP/style reward without mixing in task reward."""
    training = self.training
    self.eval()
    disc = self.forward(state, next_state)
    amp_reward = self.amp_reward_coef * self._style_score(disc)
    scale = (
      self.amp_reward_scale if amp_reward_scale is None else float(amp_reward_scale)
    )
    amp_reward *= scale
    if training:
      self.train()
    return amp_reward.squeeze(-1), disc.squeeze(-1)

  @torch.no_grad()
  def predict_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    task_reward: torch.Tensor,
    amp_reward_scale: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    amp_reward, disc = self.predict_style_reward(
      state,
      next_state,
      amp_reward_scale=amp_reward_scale,
    )
    task = task_reward.view(-1, 1)
    mixed = (1.0 - self.task_reward_lerp) * amp_reward.view(
      -1, 1
    ) + self.task_reward_lerp * task
    return mixed.squeeze(-1), disc

  def discriminator_loss(
    self,
    policy_state: torch.Tensor,
    policy_next_state: torch.Tensor,
    expert_state: torch.Tensor,
    expert_next_state: torch.Tensor,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(self.normalizer, EmpiricalNormalization):
      with torch.no_grad():
        self.normalizer.update(torch.cat((policy_state, policy_next_state), dim=-1))
        self.normalizer.update(torch.cat((expert_state, expert_next_state), dim=-1))

    policy_d = self.forward(policy_state, policy_next_state)
    expert_d = self.forward(expert_state, expert_next_state)
    if self.loss_type == "least_squares":
      expert_loss = F.mse_loss(expert_d, torch.ones_like(expert_d))
      policy_loss = F.mse_loss(policy_d, -torch.ones_like(policy_d))
    elif self.loss_type == "bce":
      expert_loss = F.binary_cross_entropy_with_logits(
        expert_d,
        torch.full_like(expert_d, self.expert_label),
      )
      policy_loss = F.binary_cross_entropy_with_logits(
        policy_d,
        torch.full_like(policy_d, self.policy_label),
      )
    else:
      raise ValueError(f"Unsupported AMP discriminator loss_type: {self.loss_type!r}.")
    amp_loss = 0.5 * (expert_loss + policy_loss)

    grad_penalty = self._gradient_penalty(expert_state, expert_next_state)
    total = amp_loss + grad_penalty
    stats = {
      "amp": amp_loss.detach(),
      "amp_grad_pen": grad_penalty.detach(),
      "amp_policy_pred": policy_d.mean().detach(),
      "amp_expert_pred": expert_d.mean().detach(),
      "amp_policy_style": self._style_score(policy_d).mean().detach(),
      "amp_expert_style": self._style_score(expert_d).mean().detach(),
    }
    return total, stats

  def _style_score(self, disc: torch.Tensor) -> torch.Tensor:
    if self.reward_type == "hardware_lsgan":
      return torch.clamp(1.0 - 0.25 * torch.square(disc - 1.0), min=0.0)
    if self.reward_type == "legacy_lsgan":
      expert_score = torch.clamp(0.5 * (disc + 1.0), min=0.0, max=1.0)
      return torch.square(expert_score)
    if self.reward_type == "sigmoid":
      return torch.sigmoid(disc)
    if self.reward_type == "softplus":
      clipped = torch.clamp(disc, min=-20.0, max=20.0)
      score = F.softplus(clipped) / F.softplus(
        torch.ones((), device=disc.device, dtype=disc.dtype)
      )
      if self.reward_clip_max > 0.0:
        score = torch.clamp(score, max=self.reward_clip_max)
      return score
    raise ValueError(f"Unsupported AMP reward_type: {self.reward_type!r}.")

  def _gradient_penalty(
    self,
    expert_state: torch.Tensor,
    expert_next_state: torch.Tensor,
  ) -> torch.Tensor:
    expert_data = torch.cat((expert_state, expert_next_state), dim=-1).detach()
    expert_data.requires_grad_(True)
    disc = self.net(self.normalizer(expert_data))
    grad = torch.autograd.grad(
      outputs=disc,
      inputs=expert_data,
      grad_outputs=torch.ones_like(disc),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return self.grad_penalty_coef * torch.square(grad.norm(2, dim=1)).mean()
