# Adapted from KaydenKnapik/BoosterT1mjlab (Apache-2.0); modified for this fork.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_lateral_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    if self.cfg.heading_command:
      self.metrics["error_heading"] += (
        torch.abs(wrap_to_pi(self.heading_target - self.robot.data.heading_w))
        / max_command_step
      )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Modes are mutually exclusive:
    # - stand: zero command
    # - x + heading: x only, y = 0, heading active
    # - lateral: y only, x = 0, heading inactive
    num_envs = len(env_ids)
    if num_envs == 0:
      return

    self.vel_command_b[env_ids] = 0.0
    self.vel_command_w[env_ids] = 0.0
    self.is_heading_env[env_ids] = False
    self.is_standing_env[env_ids] = False
    self.is_lateral_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False
    self.heading_target[env_ids] = self.robot.data.heading_w[env_ids]

    mode_sample = torch.rand(num_envs, device=self.device)
    standing_mask = mode_sample < self.cfg.rel_standing_envs
    lateral_mask = (mode_sample >= self.cfg.rel_standing_envs) & (
      mode_sample < self.cfg.rel_standing_envs + self.cfg.rel_lateral_envs
    )
    x_heading_mask = ~(standing_mask | lateral_mask)

    standing_env_ids = env_ids[standing_mask]
    lateral_env_ids = env_ids[lateral_mask]
    x_heading_env_ids = env_ids[x_heading_mask]

    self.is_standing_env[standing_env_ids] = True
    self.is_lateral_env[lateral_env_ids] = True
    self.is_world_env[env_ids] = (
      torch.rand(num_envs, device=self.device) <= self.cfg.rel_world_envs
    )

    if len(x_heading_env_ids) > 0:
      self.vel_command_b[x_heading_env_ids, 0] = torch.empty(
        len(x_heading_env_ids), device=self.device
      ).uniform_(*self.cfg.ranges.lin_vel_x)
      self.vel_command_b[x_heading_env_ids, 2] = torch.empty(
        len(x_heading_env_ids), device=self.device
      ).uniform_(*self.cfg.ranges.ang_vel_z)
      if self.cfg.heading_command:
        assert self.cfg.ranges.heading is not None
        self.heading_target[x_heading_env_ids] = torch.empty(
          len(x_heading_env_ids), device=self.device
        ).uniform_(*self.cfg.ranges.heading)
        self.is_heading_env[x_heading_env_ids] = (
          torch.rand(len(x_heading_env_ids), device=self.device)
          <= self.cfg.rel_heading_envs
        )

    if len(lateral_env_ids) > 0:
      self.vel_command_b[lateral_env_ids, 1] = torch.empty(
        len(lateral_env_ids), device=self.device
      ).uniform_(*self.cfg.ranges.lin_vel_y)

    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    fwd_candidates = x_heading_env_ids
    if len(fwd_candidates) > 0:
      self.is_forward_env[fwd_candidates] = (
        torch.rand(len(fwd_candidates), device=self.device) <= self.cfg.rel_forward_envs
      )
    fwd_ids = fwd_candidates[self.is_forward_env[fwd_candidates]]
    if len(fwd_ids) > 0:
      self.vel_command_b[fwd_ids, 0] = (
        self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
      )
      self.vel_command_b[fwd_ids, 1] = 0.0
      self.vel_command_b[fwd_ids, 2] = 0.0
      self.vel_command_w[fwd_ids] = self.vel_command_b[fwd_ids]

    init_vel_env_ids = env_ids[
      torch.rand(num_envs, device=self.device) < self.cfg.init_velocity_prob
    ]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    if self.is_world_env.any():
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      heading = self.robot.data.heading_w[w_ids]
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      vx_w = self.vel_command_w[w_ids, 0]
      vy_w = self.vel_command_w[w_ids, 1]
      self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    self.vel_command_w[standing_env_ids, :] = 0.0

  # GUI.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create velocity controls for the Viser Commands panel."""
    del on_change, request_action

    from viser import Icon

    ranges = self.cfg.ranges
    axes = [
      ("lin_vel_x", max(abs(v) for v in ranges.lin_vel_x)),
      ("lin_vel_y", max(abs(v) for v in ranges.lin_vel_y)),
    ]
    if self.cfg.heading_command:
      assert ranges.heading is not None
      axes.append(("heading_error", max(abs(v) for v in ranges.heading)))
    else:
      axes.append(("ang_vel_z", max(abs(v) for v in ranges.ang_vel_z)))

    sliders: list[viser.GuiSliderHandle] = []
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      for label, max_val in axes:
        safe_max = max(max_val, 0.1)
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=safe_max,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-safe_max,
          max=safe_max,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _slider=slider, _max_input=max_input) -> None:
          _slider.min = -_max_input.value
          _slider.max = _max_input.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for slider in sliders:
          slider.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is None or not self._joystick_enabled.value:
      return

    assert self._joystick_get_env_idx is not None
    idx = self._joystick_get_env_idx()
    self.is_standing_env[idx] = False
    self.is_lateral_env[idx] = False
    self.is_world_env[idx] = False
    self.is_forward_env[idx] = False
    self.vel_command_b[idx, 0] = self._joystick_sliders[0].value
    self.vel_command_b[idx, 1] = self._joystick_sliders[1].value
    self.vel_command_w[idx, :2] = self.vel_command_b[idx, :2]

    if self.cfg.heading_command:
      heading_error = self._joystick_sliders[2].value
      current_heading = self.robot.data.heading_w[idx]
      self.is_heading_env[idx] = True
      self.heading_error[idx] = heading_error
      self.heading_target[idx] = wrap_to_pi(current_heading + heading_error)
      self.vel_command_b[idx, 2] = torch.clamp(
        self.cfg.heading_control_stiffness
        * torch.tensor(heading_error, device=self.device),
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    else:
      self.is_heading_env[idx] = False
      self.vel_command_b[idx, 2] = self._joystick_sliders[2].value

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_lateral_envs: float = 0.0
  rel_heading_envs: float = 1.0
  rel_world_envs: float = 0.0
  rel_forward_envs: float = 0.0
  init_velocity_prob: float = 0.0

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
    mode_total = self.rel_standing_envs + self.rel_lateral_envs
    if mode_total > 1.0:
      raise ValueError(
        "The sum of rel_standing_envs and rel_lateral_envs must be <= 1.0."
      )
