"""Shared AMP observation schema for humanoid motion priors."""

from __future__ import annotations

K1_AMP_KEY_BODY_NAMES: tuple[str, ...] = (
  "left_hand_link",
  "right_hand_link",
  "left_foot_link",
  "right_foot_link",
)

K1_AMP_NUM_DOFS = 22
K1_AMP_NUM_KEY_BODIES = len(K1_AMP_KEY_BODY_NAMES)
T1_AMP_NUM_DOFS = 23
T1_AMP_KEY_BODY_NAMES = K1_AMP_KEY_BODY_NAMES
T1_AMP_NUM_KEY_BODIES = len(T1_AMP_KEY_BODY_NAMES)


def amp_obs_dim(num_dofs: int, num_key_bodies: int) -> int:
  return (
    int(num_dofs)  # joint positions
    + int(num_dofs)  # joint velocities
    + 1  # root height
    + 3  # projected gravity in root frame
    + 3  # root linear velocity in root frame
    + 3  # root angular velocity in root frame
    + 3 * int(num_key_bodies)  # key body positions in root frame
    + 3 * int(num_key_bodies)  # key body velocities in root frame
  )


K1_AMP_OBS_DIM = amp_obs_dim(K1_AMP_NUM_DOFS, K1_AMP_NUM_KEY_BODIES)
T1_AMP_OBS_DIM = amp_obs_dim(T1_AMP_NUM_DOFS, T1_AMP_NUM_KEY_BODIES)
