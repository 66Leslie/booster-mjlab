"""Goal-conditioned K1 locomotion AMP environment configuration."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_playground.amp import mdp as amp_mdp
from mjlab_playground.tasks.common import reset_joints_by_offset
from mjlab_playground.tasks.target_location_amp import mdp as target_mdp

_TARGET_POSITION_TOLERANCE = 0.16
_TARGET_YAW_TOLERANCE = 0.20
_TARGET_STABLE_HOLD_STEPS = 25
_TARGET_MAX_LIN_SPEED = 0.35
_TARGET_MAX_YAW_RATE = 0.60
_TARGET_MIN_UP_Z = 0.75


def make_target_location_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=amp_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.15, n_max=0.15),
    ),
    "projected_gravity": ObservationTermCfg(
      func=amp_mdp.projected_gravity,
      noise=Unoise(n_min=-0.04, n_max=0.04),
    ),
    "joint_pos": ObservationTermCfg(
      func=amp_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.015, n_max=0.015),
    ),
    "joint_vel": ObservationTermCfg(
      func=amp_mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=amp_mdp.last_action),
    "goal_local_pos": ObservationTermCfg(
      func=target_mdp.goal_local_position,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "heading_to_target": ObservationTermCfg(
      func=target_mdp.heading_to_target_features,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "final_yaw_error": ObservationTermCfg(
      func=target_mdp.final_yaw_error_features,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "goal_distance": ObservationTermCfg(
      func=target_mdp.goal_distance,
      scale=0.25,
    ),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(func=amp_mdp.base_lin_vel),
    "amp_robot": ObservationTermCfg(func=amp_mdp.amp_robot_obs),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.35,
    )
  }

  events = {
    "reset_robot": EventTermCfg(
      func=amp_mdp.reset_robot_near_origin,
      mode="reset",
      params={
        "robot_name": "robot",
        "x_range": (-0.04, 0.04),
        "y_range": (-0.04, 0.04),
        "yaw_range": (-0.20, 0.20),
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.01, 0.01),
        "velocity_range": (-0.02, 0.02),
      },
    ),
  }

  rewards = {
    "distance_error": RewardTermCfg(
      func=target_mdp.distance_error_penalty,
      weight=0.0,
      params={
        "command_name": "goal_pose",
        "min_distance_scale": 0.4,
      },
    ),
    "position_progress": RewardTermCfg(
      func=target_mdp.position_progress_reward,
      weight=1.75,
      params={
        "command_name": "goal_pose",
        "position_tolerance": _TARGET_POSITION_TOLERANCE,
        "yaw_tolerance": _TARGET_YAW_TOLERANCE,
        "max_speed": 1.8,
      },
    ),
    "near_goal_heading_alignment": RewardTermCfg(
      func=target_mdp.near_goal_heading_alignment_reward,
      weight=0.35,
      params={
        "command_name": "goal_pose",
        "near_distance": 0.75,
      },
    ),
    "near_goal_stability": RewardTermCfg(
      func=target_mdp.near_goal_stability_reward,
      weight=0.0,
      params={
        "command_name": "goal_pose",
        "entity_name": "robot",
        "near_distance": 0.60,
        "lin_speed_std": 0.35,
        "yaw_rate_std": 0.60,
        "min_up_z": _TARGET_MIN_UP_Z,
      },
    ),
    "yaw_progress": RewardTermCfg(
      func=target_mdp.yaw_progress_reward,
      weight=0.25,
      params={
        "command_name": "goal_pose",
        "position_tolerance": _TARGET_POSITION_TOLERANCE,
        "yaw_tolerance": _TARGET_YAW_TOLERANCE,
        "near_distance": 0.75,
      },
    ),
    "goal_position_bonus": RewardTermCfg(
      func=target_mdp.goal_position_bonus,
      weight=0.05,
      params={
        "command_name": "goal_pose",
        "position_tolerance": _TARGET_POSITION_TOLERANCE,
        "yaw_soft_tolerance": _TARGET_YAW_TOLERANCE,
      },
    ),
    "goal_pose_bonus": RewardTermCfg(
      func=target_mdp.goal_pose_terminal_bonus,
      weight=0.2,
      params={
        "command_name": "goal_pose",
        "position_tolerance": _TARGET_POSITION_TOLERANCE,
        "yaw_tolerance": _TARGET_YAW_TOLERANCE,
        "hold_steps": _TARGET_STABLE_HOLD_STEPS,
        "entity_name": "robot",
        "max_lin_speed": _TARGET_MAX_LIN_SPEED,
        "max_yaw_rate": _TARGET_MAX_YAW_RATE,
        "min_up_z": _TARGET_MIN_UP_Z,
      },
    ),
    "goal_time_efficiency": RewardTermCfg(
      func=target_mdp.goal_time_efficiency_bonus,
      weight=0.0,
      params={
        "command_name": "goal_pose",
        "position_tolerance": _TARGET_POSITION_TOLERANCE,
        "yaw_tolerance": _TARGET_YAW_TOLERANCE,
        "hold_steps": _TARGET_STABLE_HOLD_STEPS,
        "max_speed": 1.8,
        "entity_name": "robot",
        "max_lin_speed": _TARGET_MAX_LIN_SPEED,
        "max_yaw_rate": _TARGET_MAX_YAW_RATE,
        "min_up_z": _TARGET_MIN_UP_Z,
      },
    ),
    "fall_penalty": RewardTermCfg(
      func=amp_mdp.fell_over_penalty,
      weight=-0.1,
      params={
        "robot_name": "robot",
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=amp_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=amp_mdp.fell_over),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=3.0,
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=not play,
      ),
      "critic": ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
      ),
    },
    actions=actions,
    commands={
      "goal_pose": target_mdp.GoalPoseCommandCfg(
        entity_name="robot",
        resampling_time_range=(6.0, 10.0),
        debug_vis=play,
        ranges=target_mdp.GoalPoseCommandCfg.Ranges(
          target_radius=(0.4, 1.2),
          target_yaw=(-0.25, 0.25),
          target_angle=(-0.50, 0.50),
        ),
      )
    },
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics={
      "fell_over_fraction": MetricsTermCfg(
        func=amp_mdp.fell_over_fraction,
        params={"robot_name": "robot"},
      ),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso",
      distance=2.5,
      elevation=-10.0,
      azimuth=90.0,
      width=640,
      height=480,
    ),
    sim=SimulationCfg(
      nconmax=80,
      njmax=400,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=80,
        cone="elliptic",
        impratio=5.0,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
  cfg.scale_rewards_by_dt = False
  if play:
    goal_pose_cmd = cfg.commands["goal_pose"]
    assert isinstance(goal_pose_cmd, target_mdp.GoalPoseCommandCfg)
    goal_pose_cmd.ranges.target_radius = (0.4, 10.0)
    goal_pose_cmd.ranges.target_yaw = (-math.pi, math.pi)
    goal_pose_cmd.ranges.target_angle = (-math.pi, math.pi)
  return cfg
