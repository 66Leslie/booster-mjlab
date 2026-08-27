"""Observations shared by adversarial motion-prior tasks."""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_playground.amp.amp_schema import K1_AMP_KEY_BODY_NAMES

_ROBOT_CFG = SceneEntityCfg("robot", joint_names=(".*",))


def amp_robot_obs(
  env,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  key_body_names: tuple[str, ...] = K1_AMP_KEY_BODY_NAMES,
) -> torch.Tensor:
  """Build the robot-only state consumed by the AMP discriminator."""
  robot = env.scene[robot_cfg.name]
  joint_ids = robot_cfg.joint_ids
  body_ids = _resolve_key_body_ids(env, robot, key_body_names)
  root_quat = robot.data.root_link_quat_w
  root_pos = robot.data.root_link_pos_w
  root_quat_bodies = root_quat[:, None, :].expand(-1, len(body_ids), -1)

  body_pos_w = robot.data.body_link_pos_w[:, body_ids, :] - root_pos[:, None, :]
  body_vel_w = (
    robot.data.body_link_lin_vel_w[:, body_ids, :]
    - robot.data.root_link_lin_vel_w[:, None, :]
  )
  key_body_pos_b_raw = quat_apply_inverse(root_quat_bodies, body_pos_w)
  key_body_vel_b_raw = quat_apply_inverse(root_quat_bodies, body_vel_w)
  key_body_vel_b_raw = key_body_vel_b_raw - torch.cross(
    robot.data.root_link_ang_vel_b[:, None, :],
    key_body_pos_b_raw,
    dim=-1,
  )

  return torch.cat(
    (
      robot.data.joint_pos[:, joint_ids],
      robot.data.joint_vel[:, joint_ids],
      root_pos[:, 2:3],
      robot.data.projected_gravity_b,
      robot.data.root_link_lin_vel_b,
      robot.data.root_link_ang_vel_b,
      key_body_pos_b_raw.reshape(env.num_envs, -1),
      key_body_vel_b_raw.reshape(env.num_envs, -1),
    ),
    dim=-1,
  )


def _resolve_key_body_ids(env, robot, key_body_names: tuple[str, ...]) -> list[int]:
  cached_ids = getattr(env, "_amp_key_body_ids", None)
  cached_names = getattr(env, "_amp_key_body_names", None)
  if cached_ids is not None and cached_names == key_body_names:
    return cached_ids
  body_ids = robot.find_bodies(key_body_names, preserve_order=True)[0]
  if len(body_ids) != len(key_body_names):
    raise ValueError(f"Failed to resolve AMP key bodies: {key_body_names}.")
  env._amp_key_body_ids = body_ids
  env._amp_key_body_names = key_body_names
  return body_ids
