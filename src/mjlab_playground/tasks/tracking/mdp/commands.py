"""Tracking command term extensions for mixed reset sources."""

# pyright: reportIncompatibleVariableOverride=false, reportOptionalMemberAccess=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

from .events import TRACKING_HOME_RESET_MASK_KEY


class TrackingMixedResetMotionCommand(MotionCommand):
  """Motion command with optional home-pose reset on a subset of environments."""

  cfg: "TrackingMixedResetMotionCommandCfg"
  _hold_time_step_counter: torch.Tensor

  def __init__(self, cfg: "TrackingMixedResetMotionCommandCfg", env):
    super().__init__(cfg, env)
    if self.cfg.home_canonicalize_motion_yaw:
      self._canonicalize_motion_yaw_to_home()

  def _canonicalize_motion_yaw_to_home(self) -> None:
    """Rotate full motion trajectory so start-frame yaw matches home yaw.

    This removes large global heading offsets in motion files while preserving
    relative body kinematics and root pitch/roll profile.
    """
    default_root_state = self.robot.data.default_root_state
    assert default_root_state is not None
    max_frame = int(self.motion.time_step_total) - 1
    start_frame = int(max(0, min(max_frame, self.cfg.home_start_frame)))

    home_quat = default_root_state[0, 3:7].unsqueeze(0)
    source_quat = self.motion.body_quat_w[start_frame, 0].unsqueeze(0)
    home_yaw = yaw_quat(home_quat)
    source_yaw = yaw_quat(source_quat)
    delta = quat_mul(home_yaw, quat_inv(source_yaw))[0]

    delta_quat = delta.view(1, 1, 4).expand(
      self.motion.body_quat_w.shape[0], self.motion.body_quat_w.shape[1], 4
    )
    start_anchor_pos = self.motion.body_pos_w[start_frame, 0].view(1, 1, 3)

    self.motion.body_pos_w = (
      quat_apply(delta_quat, self.motion.body_pos_w - start_anchor_pos)
      + start_anchor_pos
    )
    self.motion.body_quat_w = quat_mul(delta_quat, self.motion.body_quat_w)
    self.motion.body_lin_vel_w = quat_apply(delta_quat, self.motion.body_lin_vel_w)
    self.motion.body_ang_vel_w = quat_apply(delta_quat, self.motion.body_ang_vel_w)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    """Optionally keep base scrubber behavior for interactive frame jumps.

    In play/debug for mixed-reset tracking we usually want viewer reset to preserve
    the configured reset-source policy (home vs motion), so this defaults to False.
    """
    if self.cfg.gui_reset_to_frame:
      return super().apply_gui_reset(env_ids)
    return False

  def _resample_command(self, env_ids: torch.Tensor):
    if env_ids.numel() == 0:
      return

    # Guard against invalid indices and prefer CPU-side split logic to avoid
    # opaque CUDA device-side assert when advanced indexing receives bad ids.
    env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
    min_id = int(env_ids_cpu.min().item())
    max_id = int(env_ids_cpu.max().item())
    if min_id < 0 or max_id >= self.num_envs:
      raise RuntimeError(
        f"Invalid env_ids in tracking resample: min={min_id}, max={max_id}, "
        f"num_envs={self.num_envs}"
      )

    reset_mask = self._env.extras.get(self.cfg.home_reset_mask_key)
    if (
      isinstance(reset_mask, torch.Tensor)
      and reset_mask.dtype == torch.bool
      and reset_mask.shape == (self.num_envs,)
    ):
      reset_mask_cpu = reset_mask.detach().to(device="cpu")
      home_select = reset_mask_cpu[env_ids_cpu]
      home_env_ids = env_ids_cpu[home_select].to(device=env_ids.device)
      motion_env_ids = env_ids_cpu[~home_select].to(device=env_ids.device)
    else:
      home_env_ids = env_ids.new_empty((0,), dtype=env_ids.dtype)
      motion_env_ids = env_ids

    if len(home_env_ids) > 0:
      self._resample_home_command(home_env_ids)

    # Important: MotionCommand._resample_command internally reads self.body_pos_w,
    # which indexes with self.time_steps for all envs. Therefore, any env with an
    # out-of-range timestep must be repaired before calling into the base method.
    # Home envs are repaired above; now motion envs can be safely resampled.
    if len(motion_env_ids) > 0:
      super()._resample_command(motion_env_ids)

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Write reference state with invariant or per-world joint limits."""
    soft_limits = self.robot.data.soft_joint_pos_limits
    assert soft_limits is not None
    limits = (
      soft_limits.expand(len(env_ids), -1, -1)
      if soft_limits.shape[0] == 1
      else soft_limits[env_ids]
    )
    joint_pos = torch.clip(joint_pos, limits[..., 0], limits[..., 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_home_command(self, env_ids: torch.Tensor) -> None:
    # Keep command time index fixed at the configured frame so the policy always
    # starts from the same reference timing when using home reset.
    max_frame = int(self.motion.time_step_total) - 1
    start_frame = int(max(0, min(max_frame, self.cfg.home_start_frame)))
    self.time_steps[env_ids] = start_frame

    default_root_state = self.robot.data.default_root_state
    default_joint_pos = self.robot.data.default_joint_pos
    default_joint_vel = self.robot.data.default_joint_vel
    assert default_root_state is not None
    assert default_joint_pos is not None
    assert default_joint_vel is not None

    root_pos = default_root_state[env_ids, 0:3].clone()
    root_ori = default_root_state[env_ids, 3:7].clone()
    root_lin_vel = default_root_state[env_ids, 7:10].clone()
    root_ang_vel = default_root_state[env_ids, 10:13].clone()

    pose_ranges = torch.tensor(
      [
        self.cfg.home_pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
      ],
      device=self.device,
      dtype=torch.float32,
    )
    pose_noise = sample_uniform(
      pose_ranges[:, 0],
      pose_ranges[:, 1],
      (len(env_ids), 6),
      device=self.device,
    )
    if self.cfg.home_root_position_mode == "motion_xyz":
      reference_pos = (
        self.motion.body_pos_w[start_frame, 0].unsqueeze(0).expand(len(env_ids), -1)
      )
      root_pos = reference_pos.clone()
    elif self.cfg.home_root_position_mode == "motion_xy_home_z":
      reference_pos = (
        self.motion.body_pos_w[start_frame, 0].unsqueeze(0).expand(len(env_ids), -1)
      )
      root_pos[:, :2] = reference_pos[:, :2]
    elif self.cfg.home_root_position_mode == "home":
      pass
    else:
      raise ValueError(
        f"Unsupported home_root_position_mode: {self.cfg.home_root_position_mode}"
      )
    root_pos += pose_noise[:, :3] + self._env.scene.env_origins[env_ids]
    if self.cfg.home_root_orientation_mode == "motion":
      reference_ori = (
        self.motion.body_quat_w[start_frame, 0].unsqueeze(0).expand(len(env_ids), -1)
      )
      root_ori = reference_ori.clone()
    elif self.cfg.home_root_orientation_mode == "motion_yaw":
      reference_ori = (
        self.motion.body_quat_w[start_frame, 0].unsqueeze(0).expand(len(env_ids), -1)
      )
      root_ori = yaw_quat(reference_ori)
    elif self.cfg.home_root_orientation_mode == "home":
      pass
    else:
      raise ValueError(
        f"Unsupported home_root_orientation_mode: {self.cfg.home_root_orientation_mode}"
      )
    ori_delta = quat_from_euler_xyz(
      pose_noise[:, 3],
      pose_noise[:, 4],
      pose_noise[:, 5],
    )
    root_ori = quat_mul(ori_delta, root_ori)

    vel_ranges = torch.tensor(
      [
        self.cfg.home_velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
      ],
      device=self.device,
      dtype=torch.float32,
    )
    vel_noise = sample_uniform(
      vel_ranges[:, 0],
      vel_ranges[:, 1],
      (len(env_ids), 6),
      device=self.device,
    )
    root_lin_vel += vel_noise[:, :3]
    root_ang_vel += vel_noise[:, 3:]

    joint_pos = default_joint_pos[env_ids].clone()
    joint_pos += sample_uniform(
      lower=self.cfg.home_joint_position_range[0],
      upper=self.cfg.home_joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore[arg-type]
    )

    joint_vel = default_joint_vel[env_ids].clone()
    joint_vel += sample_uniform(
      lower=self.cfg.home_joint_velocity_range[0],
      upper=self.cfg.home_joint_velocity_range[1],
      size=joint_vel.shape,
      device=joint_vel.device,  # type: ignore[arg-type]
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )

    # ManagerBasedRlEnv.reset() calls command_manager.compute(dt=0) once after
    # resampling. Hold the just-reset home frame for that first compute, and
    # optionally for extra warmup steps so policy can transition from stand/home.
    hold_steps = int(max(0, self.cfg.home_initial_hold_steps))
    if self.cfg.hold_home_frame_for_first_compute:
      hold_steps += 1
    if hold_steps > 0:
      if not hasattr(self, "_hold_time_step_counter"):
        self._hold_time_step_counter = torch.zeros(
          self.num_envs, dtype=torch.long, device=self.device
        )
      self._hold_time_step_counter[env_ids] = hold_steps

    # Mark home resets with deterministic sampling metrics.
    self.metrics["sampling_entropy"][env_ids] = 0.0
    self.metrics["sampling_top1_prob"][env_ids] = 1.0
    self.metrics["sampling_top1_bin"][env_ids] = float(start_frame) / max(
      int(self.bin_count), 1
    )

  def _update_command(self):
    hold_counter = getattr(self, "_hold_time_step_counter", None)
    if (
      isinstance(hold_counter, torch.Tensor)
      and hold_counter.dtype == torch.long
      and hold_counter.shape == (self.num_envs,)
    ):
      increment_ids = torch.where(hold_counter <= 0)[0]
      if increment_ids.numel() > 0:
        self.time_steps[increment_ids] += 1
      held_ids = torch.where(hold_counter > 0)[0]
      if held_ids.numel() > 0:
        hold_counter[held_ids] -= 1
    else:
      self.time_steps += 1

    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    self.update_relative_body_poses()

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()


@dataclass(kw_only=True)
class TrackingMixedResetMotionCommandCfg(MotionCommandCfg):
  """Configuration for mixed motion/home reset in tracking."""

  home_reset_mask_key: str = TRACKING_HOME_RESET_MASK_KEY
  home_start_frame: int = 0
  home_pose_range: dict[str, tuple[float, float]] | None = None
  home_velocity_range: dict[str, tuple[float, float]] | None = None
  home_joint_position_range: tuple[float, float] = (0.0, 0.0)
  home_joint_velocity_range: tuple[float, float] = (0.0, 0.0)
  home_root_position_mode: Literal["motion_xyz", "motion_xy_home_z", "home"] = (
    "motion_xyz"
  )
  home_root_orientation_mode: Literal["motion", "motion_yaw", "home"] = "motion"
  home_canonicalize_motion_yaw: bool = False
  gui_reset_to_frame: bool = False
  hold_home_frame_for_first_compute: bool = True
  home_initial_hold_steps: int = 0

  def build(self, env) -> TrackingMixedResetMotionCommand:
    if self.home_pose_range is None:
      self.home_pose_range = {}
    if self.home_velocity_range is None:
      self.home_velocity_range = {}
    return TrackingMixedResetMotionCommand(self, env)
