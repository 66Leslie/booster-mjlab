"""Retarget motion loaders for AMP-style priors."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab_playground.amp.amp_schema import (
  K1_AMP_KEY_BODY_NAMES,
  amp_obs_dim,
)
from mjlab_playground.asset_zoo.robots.booster_k1.k1_constants import K1_JOINT_NAMES

_DEFAULT_EMPHASIS_WEIGHT = 3.0
_DEFAULT_VELOCITY_EMPHASIS_WEIGHT = 2.0
_PACKAGED_MOTIONS_ROOT = (
  Path(__file__).resolve().parents[2] / "asset_zoo" / "motions" / "retargeted"
)
_ASSET_MOTIONS_ROOT = Path(
  os.getenv("MJLAB_PLAYGROUND_MOTION_ROOT", str(_PACKAGED_MOTIONS_ROOT))
).expanduser()


def _first_existing_dir(*candidates: Path) -> Path:
  for candidate in candidates:
    if candidate.exists():
      return candidate
  return candidates[0]


def _k1_lafan_fk_root() -> Path:
  return _ASSET_MOTIONS_ROOT / "lafan" / "k1_lafan1_fk_pkl"


def _t1_gmr_motion_root() -> Path:
  return Path(
    os.getenv(
      "MJLAB_PLAYGROUND_T1_TRACKING_MOTION_DIR",
      str(_ASSET_MOTIONS_ROOT / "tracking" / "booster_t1_23dof"),
    )
  ).expanduser()


def _t1_lafan_fk_root() -> Path:
  return _first_existing_dir(
    Path(
      os.getenv(
        "MJLAB_PLAYGROUND_T1_LOCOMOTION_MOTION_DIR",
        str(_ASSET_MOTIONS_ROOT / "lafan" / "booster_t1_lafan1_locomotion"),
      )
    ).expanduser(),
    _t1_gmr_motion_root(),
  )


def default_k1_locomotion_motion_files() -> tuple[str, ...]:
  root = _k1_lafan_fk_root()
  patterns = ("walk*.pkl", "run*.pkl", "sprint*.pkl")
  files = []
  for pattern in patterns:
    files.extend(sorted(root.glob(pattern)))
  return tuple(str(path) for path in files)


def default_k1_locomotion_support_motion_files() -> dict[str, str]:
  root = _k1_lafan_fk_root()
  return {
    "walk": str(root / "walk1_subject5.pkl"),
    "run": str(root / "run1_subject5.pkl"),
    "sprint": str(root / "sprint1_subject4.pkl"),
  }


def default_t1_locomotion_motion_files() -> tuple[str, ...]:
  root = _t1_lafan_fk_root()
  patterns = (
    "walk*.pkl",
    "run*.pkl",
    "sprint*.pkl",
    "lafan1_walk*.pkl",
    "lafan1_run*.pkl",
    "lafan1_sprint*.pkl",
  )
  files = []
  for pattern in patterns:
    files.extend(sorted(root.glob(pattern)))
  return tuple(str(path) for path in files)


def default_k1_locomotion_amp_motion_groups() -> tuple[dict[str, Any], ...]:
  return (
    {
      "name": "loco",
      "files": default_k1_locomotion_motion_files(),
      "weight": 1.0,
    },
  )


def default_t1_locomotion_amp_motion_groups() -> tuple[dict[str, Any], ...]:
  return (
    {
      "name": "loco",
      "files": default_t1_locomotion_motion_files(),
      "weight": 1.0,
    },
  )


class RetargetedAmpMotionLoader:
  """Samples consecutive humanoid AMP transitions from retarget files.

  The discriminator observes a robot-only state:
  ``joint_pos, joint_vel, root height, projected gravity, root velocity, key-body
  positions, key-body velocities``. This follows a common humanoid AMP
  convention and gives the prior visibility into end-effector and support-foot
  geometry. Key body positions come from GMR-exported
  ``local_body_pos`` fields, not an ad hoc loader-side FK pass.
  """

  def __init__(
    self,
    motion_files: tuple[str, ...] | list[str] = (),
    device: str | torch.device = "cpu",
    expected_dof: int = 22,
    motion_groups: Any | None = None,
    target_fps: float | None = None,
    joint_names: tuple[str, ...] | None = K1_JOINT_NAMES,
    key_body_names: tuple[str, ...] | None = K1_AMP_KEY_BODY_NAMES,
    emphasis_frame_offset_range: tuple[int, int] | list[int] | None = None,
    emphasis_weight: float | None = None,
    velocity_emphasis_weight: float | None = None,
  ) -> None:
    self.device = torch.device(device)
    self.expected_dof = int(expected_dof)
    self.joint_names = tuple(joint_names or K1_JOINT_NAMES)
    self.target_fps = None if target_fps is None else max(float(target_fps), 1.0e-6)
    if len(self.joint_names) != self.expected_dof:
      raise ValueError(
        "AMP joint order has "
        f"{len(self.joint_names)} names, expected {self.expected_dof}."
      )
    self.key_body_names = tuple(key_body_names or K1_AMP_KEY_BODY_NAMES)
    self.expected_amp_obs_dim = amp_obs_dim(
      self.expected_dof,
      len(self.key_body_names),
    )

    groups = self._normalize_groups(
      motion_files=motion_files,
      motion_groups=motion_groups,
      emphasis_frame_offset_range=emphasis_frame_offset_range,
      emphasis_weight=emphasis_weight,
      velocity_emphasis_weight=velocity_emphasis_weight,
    )
    if not groups:
      raise ValueError("No AMP motion files were provided.")

    loaded_groups = []
    for group in groups:
      states = []
      next_states = []
      transition_weights = []
      for motion_file in group["files"]:
        file_states, file_weights = self._load_file(
          Path(motion_file),
          emphasis_frame_offset_range=group.get("emphasis_frame_offset_range"),
          emphasis_weight=group.get("emphasis_weight", 0.0),
          velocity_emphasis_weight=group.get("velocity_emphasis_weight", 0.0),
        )
        if file_states.shape[0] < 2:
          continue
        states.append(file_states[:-1])
        next_states.append(file_states[1:])
        transition_weights.append(file_weights)
      if not states:
        continue
      sample_weights = np.concatenate(transition_weights, axis=0).astype(np.float32)
      if not np.isfinite(sample_weights).all() or np.sum(sample_weights) <= 0.0:
        sample_weights = np.ones_like(sample_weights, dtype=np.float32)
      loaded_groups.append(
        {
          "name": group["name"],
          "weight": float(group["weight"]),
          "state": torch.tensor(
            np.concatenate(states, axis=0),
            dtype=torch.float32,
            device=self.device,
          ),
          "next_state": torch.tensor(
            np.concatenate(next_states, axis=0),
            dtype=torch.float32,
            device=self.device,
          ),
          "sample_weights": torch.tensor(
            sample_weights,
            dtype=torch.float32,
            device=self.device,
          ),
        }
      )

    if not loaded_groups:
      raise ValueError("No AMP transitions were loaded from the provided motion files.")

    self._groups = loaded_groups
    self.group_names = tuple(group["name"] for group in loaded_groups)
    self.group_counts = {
      group["name"]: int(group["state"].shape[0]) for group in loaded_groups
    }
    self.group_sample_weight_sums = {
      group["name"]: float(group["sample_weights"].sum().item())
      for group in loaded_groups
    }
    self.group_weights = torch.tensor(
      [max(float(group["weight"]), 0.0) for group in loaded_groups],
      dtype=torch.float32,
      device=self.device,
    )
    if torch.sum(self.group_weights) <= 0.0:
      self.group_weights[:] = 1.0
    self.group_weights = self.group_weights / torch.sum(self.group_weights)
    self.state = torch.cat([group["state"] for group in loaded_groups], dim=0)
    self.next_state = torch.cat([group["next_state"] for group in loaded_groups], dim=0)
    self.sample_weights = torch.cat(
      [group["sample_weights"] for group in loaded_groups], dim=0
    )
    self.amp_obs_dim = int(self.state.shape[1])

  @property
  def num_transitions(self) -> int:
    return int(self.state.shape[0])

  def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(self._groups) == 1:
      group = self._groups[0]
      idx = torch.multinomial(
        group["sample_weights"],
        batch_size,
        replacement=True,
      )
      return group["state"][idx], group["next_state"][idx]

    group_ids = torch.multinomial(self.group_weights, batch_size, replacement=True)
    state = torch.empty(batch_size, self.amp_obs_dim, device=self.device)
    next_state = torch.empty_like(state)
    for group_id, group in enumerate(self._groups):
      mask = group_ids == group_id
      num = int(mask.sum().item())
      if num == 0:
        continue
      idx = torch.multinomial(group["sample_weights"], num, replacement=True)
      state[mask] = group["state"][idx]
      next_state[mask] = group["next_state"][idx]
    return state, next_state

  def _normalize_groups(
    self,
    *,
    motion_files: tuple[str, ...] | list[str],
    motion_groups: Any | None,
    emphasis_frame_offset_range: tuple[int, int] | list[int] | None,
    emphasis_weight: float | None,
    velocity_emphasis_weight: float | None,
  ) -> list[dict[str, Any]]:
    if motion_groups is None:
      files = tuple(str(path) for path in motion_files)
      default_window = _normalize_frame_offset_range(
        emphasis_frame_offset_range,
        field_name="emphasis_frame_offset_range",
      )
      default_emphasis_weight = (
        _DEFAULT_EMPHASIS_WEIGHT
        if emphasis_weight is None and default_window is not None
        else float(emphasis_weight or 0.0)
      )
      default_velocity_weight = (
        _DEFAULT_VELOCITY_EMPHASIS_WEIGHT
        if velocity_emphasis_weight is None and default_window is not None
        else float(velocity_emphasis_weight or 0.0)
      )
      return (
        [
          {
            "name": "default",
            "files": files,
            "weight": 1.0,
            "emphasis_frame_offset_range": default_window,
            "emphasis_weight": default_emphasis_weight,
            "velocity_emphasis_weight": default_velocity_weight,
          }
        ]
        if files
        else []
      )

    if isinstance(motion_groups, dict):
      iterable = [
        {"name": name, **value}
        if isinstance(value, dict)
        else {"name": name, "files": value}
        for name, value in motion_groups.items()
      ]
    else:
      iterable = list(motion_groups)

    groups = []
    for i, group in enumerate(iterable):
      if not isinstance(group, dict):
        raise TypeError(f"AMP motion group {i} must be a dict, got {type(group)!r}.")
      files = tuple(str(path) for path in group.get("files", ()))
      if not files:
        continue
      emphasis_window = _normalize_frame_offset_range(
        group.get("emphasis_frame_offset_range"),
        field_name="emphasis_frame_offset_range",
      )
      emphasis_weight = float(
        group.get(
          "emphasis_weight",
          _DEFAULT_EMPHASIS_WEIGHT if emphasis_window is not None else 0.0,
        )
      )
      velocity_weight = float(
        group.get(
          "velocity_emphasis_weight",
          _DEFAULT_VELOCITY_EMPHASIS_WEIGHT if emphasis_window is not None else 0.0,
        )
      )
      groups.append(
        {
          "name": str(group.get("name", f"group_{i}")),
          "files": files,
          "weight": float(group.get("weight", 1.0)),
          "emphasis_frame_offset_range": emphasis_window,
          "emphasis_weight": emphasis_weight,
          "velocity_emphasis_weight": velocity_weight,
        }
      )
    return groups

  def _load_file(
    self,
    path: Path,
    emphasis_frame_offset_range: tuple[int, int] | None = None,
    emphasis_weight: float = 0.0,
    velocity_emphasis_weight: float = 0.0,
  ) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
      raise FileNotFoundError(f"AMP motion file not found: {path}")
    if path.suffix.lower() == ".csv":
      return self._load_gmr_fk_pkl_for_csv(
        path,
        emphasis_frame_offset_range=emphasis_frame_offset_range,
        emphasis_weight=emphasis_weight,
        velocity_emphasis_weight=velocity_emphasis_weight,
      )
    if path.suffix.lower() not in {".pkl", ".pickle"}:
      raise ValueError(f"Unsupported AMP motion file type: {path}")
    with path.open("rb") as f:
      data = pickle.load(f)
    joint_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    fps = float(data.get("fps", 30.0))
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_quat_wxyz = _xyzw_to_wxyz(np.asarray(data["root_rot"], dtype=np.float32))
    body_names, body_pos = self._body_positions_from_pickle(data, path=path)
    root_pos, root_quat_wxyz, body_pos, joint_pos, fps = _resample_state_parts(
      root_pos,
      root_quat_wxyz,
      body_pos,
      joint_pos,
      src_fps=fps,
      target_fps=self.target_fps,
    )
    key_body_pos = self._key_body_pos_from_body_positions(
      body_names,
      body_pos,
      joint_pos=joint_pos,
      path=path,
    )
    emphasis_frame = None
    if emphasis_frame_offset_range is not None or velocity_emphasis_weight > 0.0:
      emphasis_frame = self._infer_emphasis_frame(
        body_names,
        body_pos,
        root_quat_wxyz=root_quat_wxyz,
        path=path,
      )
    transition_weights = self._transition_sample_weights(
      body_names,
      body_pos,
      num_frames=joint_pos.shape[0],
      fps=fps,
      emphasis_frame=emphasis_frame,
      emphasis_frame_offset_range=emphasis_frame_offset_range,
      emphasis_weight=emphasis_weight,
      velocity_emphasis_weight=velocity_emphasis_weight,
      path=path,
    )
    return (
      self._amp_state_obs(
        joint_pos,
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        key_body_pos_b=key_body_pos,
        fps=fps,
        path=path,
      ),
      transition_weights,
    )

  def _load_gmr_fk_pkl_for_csv(
    self,
    path: Path,
    emphasis_frame_offset_range: tuple[int, int] | None = None,
    emphasis_weight: float = 0.0,
    velocity_emphasis_weight: float = 0.0,
  ) -> tuple[np.ndarray, np.ndarray]:
    candidates = (
      path.with_suffix(".pkl"),
      path.parent.parent / f"{path.parent.name}_fk_pkl" / f"{path.stem}.pkl",
      path.parent.parent / "k1_lafan1_fk_pkl" / f"{path.stem}.pkl",
    )
    for candidate in candidates:
      if candidate.exists():
        return self._load_file(
          candidate,
          emphasis_frame_offset_range=emphasis_frame_offset_range,
          emphasis_weight=emphasis_weight,
          velocity_emphasis_weight=velocity_emphasis_weight,
        )
    raise ValueError(
      f"{path} is a root/joint CSV without GMR key-body data. AMP key-body "
      "observations require a converted pkl with local_body_pos/link_body_list. "
      f"Checked: {', '.join(str(candidate) for candidate in candidates)}."
    )

  def _amp_state_obs(
    self,
    joint_pos: np.ndarray,
    *,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    key_body_pos_b: np.ndarray,
    fps: float,
    path: Path,
  ) -> np.ndarray:
    if joint_pos.ndim != 2 or joint_pos.shape[1] != self.expected_dof:
      raise ValueError(
        f"{path} has dof_pos shape {joint_pos.shape}, expected [T, {self.expected_dof}]."
      )
    if root_pos.shape != (joint_pos.shape[0], 3):
      raise ValueError(f"{path} has root_pos shape {root_pos.shape}, expected [T, 3].")
    if root_quat_wxyz.shape != (joint_pos.shape[0], 4):
      raise ValueError(
        f"{path} has root_rot shape {root_quat_wxyz.shape}, expected [T, 4]."
      )
    if key_body_pos_b.shape != (joint_pos.shape[0], len(self.key_body_names), 3):
      raise ValueError(
        f"{path} has key-body position shape {key_body_pos_b.shape}, "
        f"expected [T, {len(self.key_body_names)}, 3]."
      )
    dt = 1.0 / max(fps, 1.0e-6)
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    root_quat_wxyz = _normalize_quat_wxyz(root_quat_wxyz)
    root_lin_vel_w = np.gradient(root_pos, dt, axis=0).astype(np.float32)
    root_lin_vel_b = _quat_apply_inverse_wxyz(root_quat_wxyz, root_lin_vel_w)
    root_ang_vel_b = _quat_to_body_ang_vel_wxyz(root_quat_wxyz, dt)
    projected_gravity_b = _quat_apply_inverse_wxyz(
      root_quat_wxyz,
      np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (joint_pos.shape[0], 1)),
    )
    key_body_vel_b = np.gradient(key_body_pos_b, dt, axis=0).astype(np.float32)
    obs = np.concatenate(
      (
        joint_pos.astype(np.float32),
        joint_vel,
        root_pos[:, 2:3].astype(np.float32),
        projected_gravity_b.astype(np.float32),
        root_lin_vel_b.astype(np.float32),
        root_ang_vel_b.astype(np.float32),
        key_body_pos_b.reshape(joint_pos.shape[0], -1).astype(np.float32),
        key_body_vel_b.reshape(joint_pos.shape[0], -1).astype(np.float32),
      ),
      axis=1,
    )
    if obs.shape[1] != self.expected_amp_obs_dim:
      raise ValueError(
        f"{path} produced AMP obs dim {obs.shape[1]}, "
        f"expected {self.expected_amp_obs_dim}."
      )
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

  def _body_positions_from_pickle(
    self,
    data: dict[str, Any],
    *,
    path: Path,
  ) -> tuple[list[str], np.ndarray]:
    if data.get("local_body_pos") is None or "link_body_list" not in data:
      raise ValueError(
        f"{path} does not contain GMR local_body_pos/link_body_list. "
        "Regenerate this motion with GMR KinematicsModel.forward_kinematics before AMP training."
      )
    body_names = list(data["link_body_list"])
    body_pos = np.asarray(data["local_body_pos"], dtype=np.float32)
    if body_pos.ndim != 3 or body_pos.shape[2] != 3:
      raise ValueError(
        f"{path} has local_body_pos shape {body_pos.shape}, expected [T, B, 3]."
      )
    return body_names, body_pos

  def _key_body_pos_from_body_positions(
    self,
    body_names: list[str],
    body_pos: np.ndarray,
    *,
    joint_pos: np.ndarray,
    path: Path,
  ) -> np.ndarray:
    ids = []
    for name in self.key_body_names:
      try:
        ids.append(body_names.index(name))
      except ValueError as exc:
        raise ValueError(f"{path} does not contain key body {name!r}.") from exc
    key_body_pos = body_pos[:, ids, :]
    if key_body_pos.shape[0] != joint_pos.shape[0]:
      raise ValueError(
        f"{path} has local_body_pos length {key_body_pos.shape[0]}, "
        f"expected {joint_pos.shape[0]}."
      )
    return key_body_pos

  def _infer_emphasis_frame(
    self,
    body_names: list[str],
    body_pos: np.ndarray,
    *,
    root_quat_wxyz: np.ndarray | None = None,
    path: Path,
  ) -> int | None:
    try:
      right_foot_id = body_names.index("right_foot_link")
    except ValueError as exc:
      raise ValueError(
        f"{path} does not contain right_foot_link in link_body_list."
      ) from exc
    right_foot_pos = body_pos[:, right_foot_id, :]
    if root_quat_wxyz is not None:
      root_roll_pitch = _roll_pitch_quat_wxyz(root_quat_wxyz)
      right_foot_pos = _quat_apply_wxyz(root_roll_pitch, right_foot_pos)
    return int(np.argmax(right_foot_pos[:, 0]))

  def _transition_sample_weights(
    self,
    body_names: list[str],
    body_pos: np.ndarray,
    *,
    num_frames: int,
    fps: float,
    emphasis_frame: int | None,
    emphasis_frame_offset_range: tuple[int, int] | None,
    emphasis_weight: float,
    velocity_emphasis_weight: float,
    path: Path,
  ) -> np.ndarray:
    transition_weights = np.ones(max(num_frames - 1, 0), dtype=np.float32)
    if transition_weights.size == 0:
      return transition_weights
    if emphasis_frame_offset_range is None and velocity_emphasis_weight <= 0.0:
      return transition_weights
    try:
      right_foot_id = body_names.index("right_foot_link")
    except ValueError as exc:
      raise ValueError(
        f"{path} does not contain right_foot_link in link_body_list."
      ) from exc
    if emphasis_frame is None:
      emphasis_frame = int(np.argmax(body_pos[:, right_foot_id, 0]))
    if emphasis_frame_offset_range is not None and emphasis_weight > 0.0:
      offset_min, offset_max = emphasis_frame_offset_range
      start = max(0, emphasis_frame + offset_min)
      stop = min(num_frames - 1, emphasis_frame + offset_max)
      transition_ids = np.arange(num_frames - 1, dtype=np.int64)
      in_window = (transition_ids >= start) & (transition_ids <= stop)
      transition_weights[in_window] += float(emphasis_weight)

    if velocity_emphasis_weight > 0.0:
      dt = 1.0 / max(float(fps), 1.0e-6)
      right_foot_vel = np.gradient(body_pos[:, right_foot_id, :], dt, axis=0)
      right_foot_speed = np.linalg.norm(right_foot_vel, axis=1)
      transition_speed = 0.5 * (right_foot_speed[:-1] + right_foot_speed[1:])
      speed_scale = float(np.percentile(transition_speed, 90.0))
      if speed_scale > 1.0e-6:
        transition_weights += float(velocity_emphasis_weight) * np.clip(
          transition_speed / speed_scale,
          0.0,
          1.0,
        ).astype(np.float32)
    return np.clip(transition_weights, 1.0e-6, None)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
  if quat_xyzw.ndim != 2 or quat_xyzw.shape[1] != 4:
    raise ValueError(f"Expected quaternion shape [T, 4], got {quat_xyzw.shape}.")
  return _normalize_quat_wxyz(quat_xyzw[:, [3, 0, 1, 2]])


def _normalize_frame_offset_range(
  value: tuple[int, int] | list[int] | None,
  *,
  field_name: str,
) -> tuple[int, int] | None:
  if value is None:
    return None
  if len(value) != 2:
    raise ValueError(f"{field_name} must have two values, got {value!r}.")
  start, stop = int(value[0]), int(value[1])
  if start > stop:
    raise ValueError(f"{field_name} start must be <= stop, got {(start, stop)!r}.")
  return start, stop


def _resample_array(
  array: np.ndarray,
  *,
  src_fps: float,
  target_fps: float | None,
) -> np.ndarray:
  array = np.asarray(array, dtype=np.float32)
  if target_fps is None or np.isclose(float(target_fps), float(src_fps)):
    return array
  if array.shape[0] <= 1:
    return array
  src_fps = max(float(src_fps), 1.0e-6)
  target_fps = max(float(target_fps), 1.0e-6)
  duration = (array.shape[0] - 1) / src_fps
  target_frames = max(int(round(duration * target_fps)) + 1, 2)
  src_t = np.arange(array.shape[0], dtype=np.float32) / src_fps
  target_t = np.linspace(0.0, duration, target_frames, dtype=np.float32)
  flat = array.reshape(array.shape[0], -1)
  resampled = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
  for dim in range(flat.shape[1]):
    resampled[:, dim] = np.interp(target_t, src_t, flat[:, dim]).astype(np.float32)
  return resampled.reshape((target_frames, *array.shape[1:]))


def _resample_state_parts(
  root_pos: np.ndarray,
  root_quat_wxyz: np.ndarray,
  body_pos: np.ndarray,
  joint_pos: np.ndarray,
  *,
  src_fps: float,
  target_fps: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
  fps = float(target_fps) if target_fps is not None else float(src_fps)
  root_pos = _resample_array(root_pos, src_fps=src_fps, target_fps=target_fps)
  root_quat = _resample_array(root_quat_wxyz, src_fps=src_fps, target_fps=target_fps)
  root_quat = _normalize_quat_wxyz(root_quat)
  body_pos = _resample_array(body_pos, src_fps=src_fps, target_fps=target_fps)
  joint_pos = _resample_array(joint_pos, src_fps=src_fps, target_fps=target_fps)
  return root_pos, root_quat, body_pos, joint_pos, fps


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
  quat = np.asarray(quat, dtype=np.float32)
  norm = np.linalg.norm(quat, axis=-1, keepdims=True)
  return quat / np.clip(norm, 1.0e-6, None)


def _quat_apply_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  quat = _normalize_quat_wxyz(quat)
  vec = np.asarray(vec, dtype=np.float32)
  if quat.ndim == vec.ndim - 1:
    quat = quat[:, None, :]
  u = quat[..., 1:]
  s = quat[..., :1]
  return vec - 2.0 * s * np.cross(u, vec) + 2.0 * np.cross(u, np.cross(u, vec))


def _quat_apply_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  quat = _normalize_quat_wxyz(quat)
  vec = np.asarray(vec, dtype=np.float32)
  if quat.ndim == vec.ndim - 1:
    quat = quat[:, None, :]
  u = quat[..., 1:]
  s = quat[..., :1]
  return (
    2.0 * np.sum(u * vec, axis=-1, keepdims=True) * u
    + (s * s - np.sum(u * u, axis=-1, keepdims=True)) * vec
    + 2.0 * s * np.cross(u, vec)
  ).astype(np.float32)


def _roll_pitch_quat_wxyz(quat: np.ndarray) -> np.ndarray:
  quat = _normalize_quat_wxyz(quat)
  w, x, y, z = np.moveaxis(quat, -1, 0)
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  half_roll = 0.5 * roll
  half_pitch = 0.5 * pitch
  cr = np.cos(half_roll)
  sr = np.sin(half_roll)
  cp = np.cos(half_pitch)
  sp = np.sin(half_pitch)
  return _normalize_quat_wxyz(
    np.stack(
      (
        cr * cp,
        sr * cp,
        cr * sp,
        -sr * sp,
      ),
      axis=-1,
    ).astype(np.float32)
  )


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    axis=-1,
  ).astype(np.float32)


def _quat_to_body_ang_vel_wxyz(quat: np.ndarray, dt: float) -> np.ndarray:
  quat = _normalize_quat_wxyz(quat)
  qdot = np.gradient(quat, dt, axis=0).astype(np.float32)
  quat_inv = quat.copy()
  quat_inv[:, 1:] *= -1.0
  omega_quat = _quat_mul_wxyz(quat_inv, qdot)
  return (2.0 * omega_quat[:, 1:]).astype(np.float32)
