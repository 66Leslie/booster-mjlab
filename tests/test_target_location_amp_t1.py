"""Tests for T1 target-location AMP task variants."""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import mjlab_playground  # noqa: F401
import mujoco
import pytest
import torch
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab_playground.tasks.target_location_amp.mdp.commands import GoalPoseCommand
from mjlab_playground.tasks.target_location_amp.mdp.state import (
  update_goal_pose_success_hold_state,
)

COMPETITION_FOOT_TASK_ID = "Mjlab-TargetLocationAmp-CompetitionFoot-Flat-Booster-T1"
COMPETITION_COLLISION_TASK_ID = (
  "Mjlab-TargetLocationAmp-CompetitionCollision-Flat-Booster-T1"
)
COMPETITION_T1_XML = Path(os.getenv("MJLAB_COMPETITION_T1_XML", ""))


def test_t1_target_location_amp_competition_foot_task_registered() -> None:
  assert COMPETITION_FOOT_TASK_ID in list_tasks()
  assert COMPETITION_COLLISION_TASK_ID in list_tasks()


def test_t1_target_location_amp_competition_foot_uses_box_contacts() -> None:
  cfg = load_env_cfg(COMPETITION_FOOT_TASK_ID)
  play_cfg = load_env_cfg(COMPETITION_FOOT_TASK_ID, play=True)
  rl_cfg = load_rl_cfg(COMPETITION_FOOT_TASK_ID)

  robot_cfg = cfg.scene.entities["robot"]
  spec = robot_cfg.spec_fn()
  foot_geoms = {geom.name: geom for geom in spec.geoms if "foot" in geom.name}

  assert "left_foot" in foot_geoms
  assert "right_foot" in foot_geoms
  assert foot_geoms["left_foot"].type == mujoco.mjtGeom.mjGEOM_BOX
  assert foot_geoms["right_foot"].type == mujoco.mjtGeom.mjGEOM_BOX
  assert foot_geoms["left_foot"].condim == 3
  assert foot_geoms["right_foot"].condim == 3
  assert tuple(foot_geoms["left_foot"].friction) == (1.0, 0.01, 0.005)
  assert tuple(foot_geoms["right_foot"].friction) == (1.0, 0.01, 0.005)
  assert tuple(foot_geoms["left_foot"].solref) == (0.02, 1.0)
  assert tuple(foot_geoms["right_foot"].solref) == (0.02, 1.0)
  assert tuple(foot_geoms["left_foot"].solimp) == (0.015, 1.0, 0.015, 0.5, 2.0)
  assert tuple(foot_geoms["right_foot"].solimp) == (0.015, 1.0, 0.015, 0.5, 2.0)
  assert not any(name.endswith("_collision") for name in foot_geoms)

  foot_friction_cfg = cfg.events["foot_friction"].params["asset_cfg"]
  assert foot_friction_cfg.geom_names == r"^(left|right)_foot$"
  assert "foot_friction" not in play_cfg.events
  assert tuple(cfg.observations["actor"].terms) == (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "goal_local_pos",
    "heading_to_target",
    "final_yaw_error",
    "goal_distance",
  )
  assert cfg.actions["joint_pos"].clip is None
  assert rl_cfg.clip_actions is None
  assert rl_cfg.experiment_name == "t1_target_location_amp_competition_foot"


def test_t1_target_location_amp_competition_variants_use_server_physics() -> None:
  for task_id in (COMPETITION_FOOT_TASK_ID, COMPETITION_COLLISION_TASK_ID):
    for play in (False, True):
      cfg = load_env_cfg(task_id, play=play)
      mujoco_cfg = cfg.sim.mujoco

      assert mujoco_cfg.integrator == "euler"
      assert mujoco_cfg.cone == "pyramidal"
      assert mujoco_cfg.impratio == 1.0
      assert mujoco_cfg.iterations == 100
      assert mujoco_cfg.ls_iterations == 50
      assert mujoco_cfg.ccd_iterations == 35


def _object_names(
  model: mujoco.MjModel, objtype: mujoco.mjtObj, count: int
) -> list[str]:
  return [mujoco.mj_id2name(model, objtype, idx) or "" for idx in range(count)]


def _exclude_body_pairs(model: mujoco.MjModel) -> set[frozenset[str]]:
  pairs: set[frozenset[str]] = set()
  for signature in model.exclude_signature:
    body_1 = int(signature) >> 16
    body_2 = int(signature) & 0xFFFF
    pairs.add(
      frozenset(
        (
          mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_1) or "",
          mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_2) or "",
        )
      )
    )
  return pairs


@pytest.mark.skipif(
  not COMPETITION_T1_XML.is_file(),
  reason="set MJLAB_COMPETITION_T1_XML to run simulator parity checks",
)
def test_t1_target_location_amp_competition_collision_matches_competition_contacts() -> (
  None
):
  cfg = load_env_cfg(COMPETITION_COLLISION_TASK_ID)
  play_cfg = load_env_cfg(COMPETITION_COLLISION_TASK_ID, play=True)
  rl_cfg = load_rl_cfg(COMPETITION_COLLISION_TASK_ID)

  robot_cfg = cfg.scene.entities["robot"]
  train_model = robot_cfg.spec_fn().compile()
  competition_model = mujoco.MjModel.from_xml_path(str(COMPETITION_T1_XML))

  train_geoms = _object_names(train_model, mujoco.mjtObj.mjOBJ_GEOM, train_model.ngeom)
  competition_geoms = _object_names(
    competition_model, mujoco.mjtObj.mjOBJ_GEOM, competition_model.ngeom
  )
  train_collision_geoms = {
    name
    for idx, name in enumerate(train_geoms)
    if train_model.geom_contype[idx] != 0 or train_model.geom_conaffinity[idx] != 0
  }
  competition_collision_geoms = {
    name
    for idx, name in enumerate(competition_geoms)
    if (
      competition_model.geom_contype[idx] != 0
      or competition_model.geom_conaffinity[idx] != 0
    )
  }
  assert train_collision_geoms == competition_collision_geoms
  assert not any(name.endswith("_collision") for name in train_collision_geoms)
  assert _exclude_body_pairs(train_model) == _exclude_body_pairs(competition_model)

  for geom_name in sorted(competition_collision_geoms):
    train_idx = train_geoms.index(geom_name)
    competition_idx = competition_geoms.index(geom_name)
    assert (
      train_model.geom_type[train_idx] == competition_model.geom_type[competition_idx]
    )
    assert (
      train_model.geom_condim[train_idx]
      == competition_model.geom_condim[competition_idx]
    )
    assert (
      train_model.geom_contype[train_idx]
      == competition_model.geom_contype[competition_idx]
    )
    assert (
      train_model.geom_conaffinity[train_idx]
      == competition_model.geom_conaffinity[competition_idx]
    )
    assert tuple(train_model.geom_size[train_idx]) == tuple(
      competition_model.geom_size[competition_idx]
    )
    assert tuple(train_model.geom_pos[train_idx]) == tuple(
      competition_model.geom_pos[competition_idx]
    )
    assert tuple(train_model.geom_friction[train_idx]) == tuple(
      competition_model.geom_friction[competition_idx]
    )
    assert tuple(train_model.geom_solref[train_idx]) == tuple(
      competition_model.geom_solref[competition_idx]
    )
    assert tuple(train_model.geom_solimp[train_idx]) == tuple(
      competition_model.geom_solimp[competition_idx]
    )

  train_bodies = _object_names(train_model, mujoco.mjtObj.mjOBJ_BODY, train_model.nbody)
  competition_bodies = _object_names(
    competition_model, mujoco.mjtObj.mjOBJ_BODY, competition_model.nbody
  )
  for body_name in ("left_foot_link", "right_foot_link"):
    train_idx = train_bodies.index(body_name)
    competition_idx = competition_bodies.index(body_name)
    assert tuple(train_model.body_inertia[train_idx]) == tuple(
      competition_model.body_inertia[competition_idx]
    )
    assert tuple(train_model.body_iquat[train_idx]) == tuple(
      competition_model.body_iquat[competition_idx]
    )

  assert all(
    actuator.viscous_damping == 0.01 for actuator in robot_cfg.articulation.actuators
  )

  foot_friction_cfg = cfg.events["foot_friction"].params["asset_cfg"]
  assert foot_friction_cfg.geom_names == r"^(left|right)_foot$"
  assert "foot_friction" not in play_cfg.events
  assert tuple(cfg.observations["actor"].terms) == (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "goal_local_pos",
    "heading_to_target",
    "final_yaw_error",
    "goal_distance",
  )
  assert cfg.actions["joint_pos"].clip is None
  assert rl_cfg.clip_actions is None
  assert rl_cfg.experiment_name == "t1_target_location_amp_competition_collision"


def test_t1_target_location_amp_competition_collision_play_uses_nearfield_stage() -> (
  None
):
  play_cfg = load_env_cfg(COMPETITION_COLLISION_TASK_ID, play=True)
  goal_pose_cmd = play_cfg.commands["goal_pose"]

  assert goal_pose_cmd.ranges.target_radius == (0.0, 6.0)
  assert goal_pose_cmd.ranges.target_yaw == (-math.pi, math.pi)
  assert goal_pose_cmd.ranges.target_angle == (-math.pi, math.pi)
  assert goal_pose_cmd.ranges.target_radius_buckets is None
  assert goal_pose_cmd.ranges.target_command_buckets is not None
  assert tuple(
    (bucket.radius_range, bucket.yaw_range, bucket.angle_range, bucket.probability)
    for bucket in goal_pose_cmd.ranges.target_command_buckets
  ) == (
    ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 0.1),
    ((0.0, 0.0), (-math.pi, math.pi), (0.0, 0.0), 0.1),
    ((0.0, 2.0), None, None, 0.3),
    ((5.0, 6.0), None, None, 0.4),
    ((2.0, 4.0), None, None, 0.1),
  )
  assert goal_pose_cmd.resampling_time_range == (1.0, 4.0)
  assert goal_pose_cmd.resample_on_near_goal is False
  assert goal_pose_cmd.resample_on_goal_success is True
  assert math.isclose(goal_pose_cmd.resample_position_tolerance, 0.08)
  assert math.isclose(goal_pose_cmd.resample_yaw_tolerance, math.radians(6.0))

  assert play_cfg.rewards["joint_target_limits"].weight == -0.03
  assert play_cfg.rewards["near_goal_stability"].weight == 0.5
  assert math.isclose(
    play_cfg.rewards["near_goal_stability"].params["position_tolerance"],
    0.08,
  )
  assert math.isclose(
    play_cfg.rewards["near_goal_stability"].params["yaw_tolerance"],
    math.radians(6.0),
  )
  assert play_cfg.rewards["goal_pose_bonus"].weight == 1.0
  assert play_cfg.rewards["goal_position_bonus"].weight == 0.5
  assert math.isclose(
    play_cfg.rewards["goal_pose_bonus"].params["position_tolerance"], 0.08
  )
  assert math.isclose(
    play_cfg.rewards["goal_pose_bonus"].params["yaw_tolerance"],
    math.radians(6.0),
  )
  assert play_cfg.rewards["goal_pose_bonus"].params["hold_steps"] == 25


def test_goal_pose_command_can_resample_early_on_success() -> None:
  reward_manager = SimpleNamespace(
    get_term_cfg=lambda name: SimpleNamespace(
      params={
        "position_tolerance": 0.08,
        "yaw_tolerance": math.radians(6.0),
        "hold_steps": 25,
        "entity_name": "robot",
        "max_lin_speed": 0.25,
        "max_yaw_rate": 0.35,
        "min_up_z": 0.80,
      }
      if name == "goal_pose_bonus"
      else None
    )
  )
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.7]], dtype=torch.float32),
      root_link_lin_vel_b=torch.zeros(1, 3, dtype=torch.float32),
      root_link_ang_vel_b=torch.zeros(1, 3, dtype=torch.float32),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
    )
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    reward_manager=reward_manager,
    scene={"robot": robot},
    episode_length_buf=torch.tensor([1], dtype=torch.long),
  )
  command = GoalPoseCommand.__new__(GoalPoseCommand)
  command._env = env
  command.cfg = SimpleNamespace(
    entity_name="robot",
    resample_on_goal_success=True,
    resample_on_near_goal=False,
    resample_position_tolerance=0.08,
    resample_yaw_tolerance=math.radians(6.0),
  )
  command.distance_to_goal = torch.tensor([0.02], dtype=torch.float32)
  command.final_heading_error = torch.tensor([math.radians(2.0)], dtype=torch.float32)
  command._resample_on_next_step = torch.zeros(1, dtype=torch.bool)

  command._queue_goal_success_resample()
  assert command._resample_on_next_step.tolist() == [False]

  env._goal_pose_prev_distance = torch.tensor([0.02], dtype=torch.float32)
  env._goal_pose_prev_yaw_abs = torch.tensor([math.radians(2.0)], dtype=torch.float32)
  env._goal_pose_distance_progress = torch.zeros(1, dtype=torch.float32)
  env._goal_pose_yaw_progress = torch.zeros(1, dtype=torch.float32)
  env._goal_pose_hold_steps = torch.tensor([24], dtype=torch.long)
  env._goal_pose_success_hold_steps = torch.tensor([24], dtype=torch.long)
  env._goal_pose_success_hold_last_step = torch.tensor([24], dtype=torch.long)
  env._goal_pose_state_last_step = torch.tensor([24], dtype=torch.long)
  env._goal_pose_just_resampled = torch.tensor([False], dtype=torch.bool)
  env.episode_length_buf = torch.tensor([25], dtype=torch.long)
  command._queue_goal_success_resample()
  assert command._resample_on_next_step.tolist() == [True]


def test_goal_success_hold_requires_consecutive_full_success_steps() -> None:
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    episode_length_buf=torch.tensor([1], dtype=torch.long),
  )

  hold = update_goal_pose_success_hold_state(
    env,
    success_now=torch.tensor([True]),
  )
  assert hold.tolist() == [1]

  env.episode_length_buf[:] = 2
  hold = update_goal_pose_success_hold_state(
    env,
    success_now=torch.tensor([False]),
  )
  assert hold.tolist() == [0]

  env.episode_length_buf[:] = 3
  hold = update_goal_pose_success_hold_state(
    env,
    success_now=torch.tensor([True]),
  )
  assert hold.tolist() == [1]

  # Multiple reward/command consumers in one controller step must not double count.
  hold = update_goal_pose_success_hold_state(
    env,
    success_now=torch.tensor([True]),
  )
  assert hold.tolist() == [1]
