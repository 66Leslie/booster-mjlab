"""Booster K1 getup environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_playground.asset_zoo.robots.booster_k1.k1_constants import get_k1_robot_cfg
from mjlab_playground.tasks.getup import mdp
from mjlab_playground.tasks.getup.getup_env_cfg import make_getup_env_cfg
from mjlab_playground.tasks.getup.mdp.actions import (
  SettleRelativeJointPositionActionCfg,
)

# Derived from the K1 standing/getup calibration in the MotrixLab scene.
_TORSO_HEIGHT = 0.543
_HIP_HEIGHT = 0.466


def booster_k1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 getup task configuration."""
  cfg = make_getup_env_cfg()

  cfg.scene.entities = {"robot": get_k1_robot_cfg()}
  cfg.episode_length_s = 8.0

  cfg.observations["actor"].terms["base_ang_vel"].func = mdp.base_ang_vel
  cfg.observations["actor"].terms["base_ang_vel"].params = {}
  cfg.observations["critic"].terms["base_ang_vel"].func = mdp.base_ang_vel
  cfg.observations["critic"].terms["base_ang_vel"].params = {}
  cfg.observations["critic"].terms["base_lin_vel"].func = mdp.base_lin_vel
  cfg.observations["critic"].terms["base_lin_vel"].params = {}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (self_collision_cfg,)

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )

  cfg.rewards["torso_height"].params["desired_height"] = _TORSO_HEIGHT
  cfg.rewards["torso_height"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("torso",)
  )
  cfg.rewards["left_hip_height"] = RewardTermCfg(
    func=mdp.height_reward,
    weight=0.5,
    params={
      "desired_height": _HIP_HEIGHT,
      "asset_cfg": SceneEntityCfg("robot", body_names=("Left_Hip_Pitch",)),
    },
  )
  cfg.rewards["right_hip_height"] = RewardTermCfg(
    func=mdp.height_reward,
    weight=0.5,
    params={
      "desired_height": _HIP_HEIGHT,
      "asset_cfg": SceneEntityCfg("robot", body_names=("Right_Hip_Pitch",)),
    },
  )
  cfg.metrics["getup_success"].params["desired_height"] = _TORSO_HEIGHT

  cfg.rewards["posture"].params["std"] = {
    r".*_Hip_Roll": 0.08,
    r".*_Hip_Yaw": 0.08,
    r".*_Hip_Pitch": 0.12,
    r".*_Knee_Pitch": 0.15,
    r".*_Ankle_Pitch": 0.2,
    r".*_Ankle_Roll": 0.2,
    r"(AAHead_yaw|Head_pitch)": 0.15,
    r"(.*_Shoulder.*|.*_Elbow.*)": 0.5,
  }

  cfg.viewer.body_name = "torso"

  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("torso",)
  )

  cfg.events["reset_fallen_or_standing"].params["fall_height"] = 0.65

  assert isinstance(cfg.actions["joint_pos"], SettleRelativeJointPositionActionCfg)
  cfg.actions["joint_pos"].settle_steps = 50
  cfg.terminations["energy"].params["settle_steps"] = 50
  cfg.terminations["energy"].params["threshold"] = float("inf")
  cfg.rewards["action_rate_l2"].weight = -0.005
  cfg.rewards["joint_vel_l2"].weight = 0.0

  # K1 has no waist joint, so supine recovery needs larger full-body swings for
  # longer than T1. Keep the getup reward simple, but delay regularization.
  cfg.curriculum = {
    "action_rate_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "action_rate_l2",
        "stages": [
          {"step": 0, "weight": -0.005},
          {"step": 1500 * 24, "weight": -0.01},
          {"step": 2500 * 24, "weight": -0.03},
          {"step": 3500 * 24, "weight": -0.05},
          {"step": 4500 * 24, "weight": -0.08},
        ],
      },
    ),
    "joint_vel_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_l2",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": 2500 * 24, "weight": -0.002},
          {"step": 3500 * 24, "weight": -0.005},
          {"step": 4500 * 24, "weight": -0.008},
        ],
      },
    ),
  }

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.events["reset_fallen_or_standing"].params["fall_probability"] = 1.0

  return cfg
