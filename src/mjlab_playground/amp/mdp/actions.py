"""Action terms shared by dynamic AMP tasks."""

# pyright: reportIncompatibleVariableOverride=false

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
  """Joint position control with policy-step action application latency."""

  delay_steps: int = 0
  """Number of controller steps between commanding and applying an action."""

  startup_hold_steps: int = 0
  """Episode-start steps that hold the neutral pose before policy takeover."""

  def build(self, env: ManagerBasedRlEnv) -> DelayedJointPositionAction:
    return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
  """Delay actuator targets without changing ActionManager action history.

  The policy-visible current/previous actions remain the latest commanded raw
  actions. Only the processed position target written to MuJoCo is delayed,
  matching rcssservermj's parallel observation/action loop.
  """

  cfg: DelayedJointPositionActionCfg

  def __init__(
    self,
    cfg: DelayedJointPositionActionCfg,
    env: ManagerBasedRlEnv,
  ) -> None:
    if cfg.delay_steps < 0:
      raise ValueError(f"delay_steps must be non-negative, got {cfg.delay_steps}.")
    if cfg.startup_hold_steps < 0:
      raise ValueError(
        f"startup_hold_steps must be non-negative, got {cfg.startup_hold_steps}."
      )
    super().__init__(cfg=cfg, env=env)
    self._delay_steps = int(cfg.delay_steps)
    self._startup_hold_steps = int(cfg.startup_hold_steps)
    neutral = self._neutral_processed_action()
    self._applied_processed_actions = neutral.clone()
    self._processed_action_history = [neutral.clone() for _ in range(self._delay_steps)]

  @property
  def applied_processed_action(self) -> torch.Tensor:
    """Processed target currently written to the simulated actuators."""
    return self._applied_processed_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    if self._startup_hold_steps > 0:
      actions = actions.clone()
      hold_mask = self._env.episode_length_buf < self._startup_hold_steps
      actions[hold_mask] = 0.0
    super().process_actions(actions)
    if self._delay_steps == 0:
      self._applied_processed_actions = self._processed_actions
      return

    self._applied_processed_actions = self._processed_action_history.pop(0)
    self._processed_action_history.append(self._processed_actions.clone())

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._applied_processed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids=env_ids)
    neutral = self._neutral_processed_action()
    self._processed_actions[env_ids] = neutral[env_ids]
    self._applied_processed_actions[env_ids] = neutral[env_ids]
    for history_entry in self._processed_action_history:
      history_entry[env_ids] = neutral[env_ids]

  def _neutral_processed_action(self) -> torch.Tensor:
    neutral = torch.zeros_like(self._processed_actions) * self._scale + self._offset
    if self.cfg.clip is not None:
      neutral = torch.clamp(
        neutral,
        min=self._clip[:, :, 0],
        max=self._clip[:, :, 1],
      )
    return neutral
