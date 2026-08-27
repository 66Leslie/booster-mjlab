"""Booster T1 constants.

This profile keeps the mjlab position-actuator control law but aligns the
actuated joint limits/passive terms with the active competition-side T1:
- joint armature: 0.01
- joint viscous damping: 0.01 for the competition-collision profile
- joint frictionloss: 0.1
- actuator effort limits from the competition robot joint classes
"""

# pyright: reportAttributeAccessIssue=false

from copy import deepcopy
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

T1_XML: Path = Path(__file__).parent / "xmls" / "t1.xml"
assert T1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(T1_XML))


##
# Actuator config.
# Reference: https://booster.feishu.cn/wiki/JGZAwk8CUi5m6nklgxMcp2KlnVe
##

_rpm = lambda r: r * 2 * 3.14159265 / 60  # noqa: E731

NECK_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(18e-6, 10),
  velocity_limit=_rpm(400),
  effort_limit=7.0,
)

ARM_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(21.8e-6, 36),
  velocity_limit=_rpm(89),
  effort_limit=18.0,
)

WAIST_HIP_ROLL_YAW_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(76.5e-6, 25),
  velocity_limit=_rpm(70),
  effort_limit=30.0,
)

HIP_PITCH_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(161.7e-6, 18),
  velocity_limit=_rpm(157),
  effort_limit=45.0,
)

KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(196.3e-6, 18),
  velocity_limit=_rpm(140),
  effort_limit=60.0,
)

ANKLE_PITCH_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(26.2e-6, 36),
  velocity_limit=_rpm(117),
  effort_limit=20.0,
)

ANKLE_ROLL_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(26.2e-6, 36),
  velocity_limit=_rpm(117),
  effort_limit=15.0,
)

NATURAL_FREQ = 5.0 * 2.0 * 3.14159265  # 5 Hz
DAMPING_RATIO = 2.0
COMPETITION_JOINT_ARMATURE = 0.01
COMPETITION_JOINT_FRICTIONLOSS = 0.1
COMPETITION_JOINT_VISCOUS_DAMPING = 0.01

T1_JOINT_NAMES: tuple[str, ...] = (
  "AAHead_yaw",
  "Head_pitch",
  "Left_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
  "Waist",
  "Left_Hip_Pitch",
  "Left_Hip_Roll",
  "Left_Hip_Yaw",
  "Left_Knee_Pitch",
  "Left_Ankle_Pitch",
  "Left_Ankle_Roll",
  "Right_Hip_Pitch",
  "Right_Hip_Roll",
  "Right_Hip_Yaw",
  "Right_Knee_Pitch",
  "Right_Ankle_Pitch",
  "Right_Ankle_Roll",
)


def _kp(act: ElectricActuator) -> float:
  return act.reflected_inertia * NATURAL_FREQ**2


def _kv(act: ElectricActuator) -> float:
  return 2.0 * DAMPING_RATIO * act.reflected_inertia * NATURAL_FREQ


T1_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
  target_names_expr=("AAHead_yaw", "Head_pitch"),
  stiffness=_kp(NECK_ACTUATOR),
  damping=_kv(NECK_ACTUATOR),
  effort_limit=NECK_ACTUATOR.effort_limit,
  armature=COMPETITION_JOINT_ARMATURE,
  frictionloss=COMPETITION_JOINT_FRICTIONLOSS,
)

T1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_Shoulder_Pitch",
    ".*_Shoulder_Roll",
    ".*_Elbow_Pitch",
    ".*_Elbow_Yaw",
  ),
  stiffness=_kp(ARM_ACTUATOR),
  damping=_kv(ARM_ACTUATOR),
  effort_limit=ARM_ACTUATOR.effort_limit,
  armature=COMPETITION_JOINT_ARMATURE,
  frictionloss=COMPETITION_JOINT_FRICTIONLOSS,
)

T1_ACTUATOR_WAIST_HIP_ROLL_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("Waist", ".*_Hip_Roll", ".*_Hip_Yaw"),
  stiffness=_kp(WAIST_HIP_ROLL_YAW_ACTUATOR),
  damping=_kv(WAIST_HIP_ROLL_YAW_ACTUATOR),
  effort_limit=WAIST_HIP_ROLL_YAW_ACTUATOR.effort_limit,
  armature=COMPETITION_JOINT_ARMATURE,
  frictionloss=COMPETITION_JOINT_FRICTIONLOSS,
)

T1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Pitch",),
  stiffness=_kp(HIP_PITCH_ACTUATOR),
  damping=_kv(HIP_PITCH_ACTUATOR),
  effort_limit=HIP_PITCH_ACTUATOR.effort_limit,
  armature=COMPETITION_JOINT_ARMATURE,
  frictionloss=COMPETITION_JOINT_FRICTIONLOSS,
)

T1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Knee_Pitch",),
  stiffness=_kp(KNEE_ACTUATOR),
  damping=_kv(KNEE_ACTUATOR),
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=COMPETITION_JOINT_ARMATURE,
  frictionloss=COMPETITION_JOINT_FRICTIONLOSS,
)

T1_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Ankle_Pitch",),
  stiffness=_kp(ANKLE_PITCH_ACTUATOR),
  damping=_kv(ANKLE_PITCH_ACTUATOR),
  effort_limit=ANKLE_PITCH_ACTUATOR.effort_limit,
  armature=COMPETITION_JOINT_ARMATURE,
  frictionloss=COMPETITION_JOINT_FRICTIONLOSS,
)

T1_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Ankle_Roll",),
  stiffness=_kp(ANKLE_ROLL_ACTUATOR),
  damping=_kv(ANKLE_ROLL_ACTUATOR),
  effort_limit=ANKLE_ROLL_ACTUATOR.effort_limit,
  armature=COMPETITION_JOINT_ARMATURE,
  frictionloss=COMPETITION_JOINT_FRICTIONLOSS,
)

##
# Keyframes.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.665),
  joint_pos={
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Yaw": 0.4,
    ".*_Hip_Pitch": -0.2,
    ".*_Knee_Pitch": 0.4,
    ".*_Ankle_Pitch": -0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^(left|right)_foot\d+_collision$"
_competition_foot_regex = r"^(left|right)_foot$"
_competition_foot_friction = (1.0, 0.01, 0.005)
_competition_foot_solref = (0.02, 1.0)
_competition_foot_solimp = (0.015, 1.0, 0.015, 0.5, 2.0)
_competition_collision_regex = (
  r"^(torso|head|left_forearm|left_hand|right_forearm|right_hand|"
  r"left_thigh|left_calf|left_foot|right_thigh|right_calf|right_foot)$"
)
_competition_collision_friction = (1.0, 0.01, 0.005)
_competition_collision_solref = (0.02, 1.0)
_competition_collision_solimp = (0.9, 0.95, 0.001, 0.5, 2.0)
_competition_collision_rgba = (0.0, 1.0, 0.0, 0.0)
_competition_foot_inertia = (0.00268212, 0.002385, 0.000726885)
_competition_foot_iquat = (0.0, 0.6799274021193834, 0.0, 0.7332794336725845)
_competition_contact_excludes = (
  ("Hip_Roll_Left", "Shank_Left"),
  ("Shank_Left", "left_foot_link"),
  ("Hip_Roll_Right", "Shank_Right"),
  ("Shank_Right", "right_foot_link"),
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  solref=(0.01, 1),
  condim={_foot_regex: 6, ".*_collision": 3},
  friction={_foot_regex: (1, 5e-3, 5e-4), ".*_collision": (0.6,)},
  priority=1,
)

COMPETITION_FOOT_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", _competition_foot_regex),
  solref={_competition_foot_regex: _competition_foot_solref, ".*_collision": (0.01, 1)},
  solimp={_competition_foot_regex: _competition_foot_solimp},
  condim={_competition_foot_regex: 3, ".*_collision": 3},
  friction={
    _competition_foot_regex: _competition_foot_friction,
    ".*_collision": (0.6,),
  },
  priority=1,
)

COMPETITION_COLLISION = CollisionCfg(
  geom_names_expr=(_competition_collision_regex,),
  solref=_competition_collision_solref,
  solimp={
    _competition_foot_regex: _competition_foot_solimp,
    _competition_collision_regex: _competition_collision_solimp,
  },
  condim=3,
  friction=_competition_collision_friction,
  priority=1,
)

##
# Final config.
##

T1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    T1_ACTUATOR_NECK,
    T1_ACTUATOR_ARM,
    T1_ACTUATOR_WAIST_HIP_ROLL_YAW,
    T1_ACTUATOR_HIP_PITCH,
    T1_ACTUATOR_KNEE,
    T1_ACTUATOR_ANKLE_PITCH,
    T1_ACTUATOR_ANKLE_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_t1_robot_cfg() -> EntityCfg:
  """Get a fresh T1 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=T1_ARTICULATION,
  )


def get_competition_foot_spec() -> mujoco.MjSpec:
  """Get the T1 spec with competition-style box foot collision geoms.

  This keeps the training-side T1 meshes, inertials, joints, and actuators, but
  replaces the four capsule foot collision geoms per foot with one box collision
  geom per foot matching the competition T1 foot shape.
  """
  spec = get_spec()
  capsule_foot_names = {
    f"{side}_foot{idx}_collision" for side in ("left", "right") for idx in range(1, 5)
  }
  for geom in tuple(spec.geoms):
    if geom.name in capsule_foot_names:
      spec.delete(geom)

  for side in ("left", "right"):
    body = spec.body(f"{side}_foot_link")
    body.add_geom(
      name=f"{side}_foot",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=(0.01, 0.0, -0.015),
      size=(0.1115, 0.05, 0.015),
      contype=1,
      conaffinity=1,
      condim=3,
      priority=1,
      friction=_competition_foot_friction,
      solref=_competition_foot_solref,
      solimp=_competition_foot_solimp,
      rgba=(0.0, 1.0, 0.0, 0.5),
    )
  return spec


_T1_TRAINING_COLLISION_GEOM_NAMES = {
  "trunk_collision",
  "head_collision",
  "left_shoulder_collision",
  "left_upper_arm_collision",
  "left_elbow_collision",
  "left_lower_arm_collision",
  "right_shoulder_collision",
  "right_upper_arm_collision",
  "right_elbow_collision",
  "right_lower_arm_collision",
  "waist_collision",
  "left_thigh_collision",
  "left_shin_collision",
  "left_knee_collision",
  "right_thigh_collision",
  "right_shin_collision",
  "right_knee_collision",
  *{f"{side}_foot{idx}_collision" for side in ("left", "right") for idx in range(1, 5)},
}


def _delete_named_geoms(spec: mujoco.MjSpec, geom_names: set[str]) -> None:
  for geom in tuple(spec.geoms):
    if geom.name in geom_names:
      spec.delete(geom)


def _add_competition_collision_geom(
  spec: mujoco.MjSpec,
  *,
  body_name: str,
  geom_name: str,
  geom_type: mujoco.mjtGeom,
  pos: tuple[float, float, float],
  size: tuple[float, float, float],
  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
  solimp: tuple[float, float, float, float, float] = _competition_collision_solimp,
) -> None:
  spec.body(body_name).add_geom(
    name=geom_name,
    type=geom_type,
    pos=pos,
    size=size,
    quat=quat,
    contype=1,
    conaffinity=1,
    condim=3,
    group=3,
    priority=1,
    friction=_competition_collision_friction,
    solref=_competition_collision_solref,
    solimp=solimp,
    rgba=_competition_collision_rgba,
  )


def get_competition_collision_spec() -> mujoco.MjSpec:
  """Get the T1 spec with competition-style collision geoms.

  The source remains the training T1 MJCF so body/joint names used by the
  mjlab task stay stable. Only collision geoms and the foot inertial frame are
  aligned to the active competition T1 XML.
  """
  spec = get_spec()
  _delete_named_geoms(spec, _T1_TRAINING_COLLISION_GEOM_NAMES)

  # The competition server excludes different non-adjacent self-contact pairs
  # than the original training MJCF. These pairs materially change leg motion.
  for exclude in tuple(spec.excludes):
    spec.delete(exclude)
  for body_name_1, body_name_2 in _competition_contact_excludes:
    spec.add_exclude(bodyname1=body_name_1, bodyname2=body_name_2)

  for side in ("left", "right"):
    foot_body = spec.body(f"{side}_foot_link")
    foot_body.inertia = _competition_foot_inertia
    foot_body.iquat = _competition_foot_iquat

  _add_competition_collision_geom(
    spec,
    body_name="Trunk",
    geom_name="torso",
    geom_type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.06, 0.0, 0.12),
    size=(0.075, 0.1, 0.15),
  )
  _add_competition_collision_geom(
    spec,
    body_name="H2",
    geom_name="head",
    geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
    pos=(0.01, 0.0, 0.11),
    size=(0.08, 0.0, 0.0),
  )
  _add_competition_collision_geom(
    spec,
    body_name="AL3",
    geom_name="left_forearm",
    geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    pos=(0.0, 0.05, 0.0),
    size=(0.03, 0.075, 0.0),
    quat=(0.70710678, 0.70710678, 0.0, 0.0),
  )
  _add_competition_collision_geom(
    spec,
    body_name="left_hand_link",
    geom_name="left_hand",
    geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    pos=(0.0, 0.13, 0.0),
    size=(0.03, 0.0875, 0.0),
    quat=(0.70710678, 0.70710678, 0.0, 0.0),
  )
  _add_competition_collision_geom(
    spec,
    body_name="AR3",
    geom_name="right_forearm",
    geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    pos=(0.0, -0.05, 0.0),
    size=(0.03, 0.075, 0.0),
    quat=(0.70710678, 0.70710678, 0.0, 0.0),
  )
  _add_competition_collision_geom(
    spec,
    body_name="right_hand_link",
    geom_name="right_hand",
    geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    pos=(0.0, -0.13, 0.0),
    size=(0.03, 0.0875, 0.0),
    quat=(0.70710678, 0.70710678, 0.0, 0.0),
  )
  _add_competition_collision_geom(
    spec,
    body_name="Hip_Roll_Left",
    geom_name="left_thigh",
    geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    pos=(0.0, 0.0, -0.08),
    size=(0.05, 0.08, 0.0),
  )
  _add_competition_collision_geom(
    spec,
    body_name="Shank_Left",
    geom_name="left_calf",
    geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    pos=(0.0, 0.0, -0.12),
    size=(0.05, 0.075, 0.0),
  )
  _add_competition_collision_geom(
    spec,
    body_name="left_foot_link",
    geom_name="left_foot",
    geom_type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.01, 0.0, -0.015),
    size=(0.1115, 0.05, 0.015),
    solimp=_competition_foot_solimp,
  )
  _add_competition_collision_geom(
    spec,
    body_name="Hip_Roll_Right",
    geom_name="right_thigh",
    geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    pos=(0.0, 0.0, -0.08),
    size=(0.05, 0.08, 0.0),
  )
  _add_competition_collision_geom(
    spec,
    body_name="Shank_Right",
    geom_name="right_calf",
    geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    pos=(0.0, 0.0, -0.12),
    size=(0.05, 0.075, 0.0),
  )
  _add_competition_collision_geom(
    spec,
    body_name="right_foot_link",
    geom_name="right_foot",
    geom_type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.01, 0.0, -0.015),
    size=(0.1115, 0.05, 0.015),
    solimp=_competition_foot_solimp,
  )
  return spec


def get_t1_competition_foot_robot_cfg() -> EntityCfg:
  """Get a fresh T1 robot config with competition-style box foot contacts."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(COMPETITION_FOOT_COLLISION,),
    spec_fn=get_competition_foot_spec,
    articulation=T1_ARTICULATION,
  )


def get_t1_competition_collision_robot_cfg() -> EntityCfg:
  """Get a fresh T1 robot config with competition-style collision contacts."""
  articulation = deepcopy(T1_ARTICULATION)
  for actuator in articulation.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    actuator.viscous_damping = COMPETITION_JOINT_VISCOUS_DAMPING
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(COMPETITION_COLLISION,),
    spec_fn=get_competition_collision_spec,
    articulation=articulation,
  )


T1_ACTION_SCALE: dict[str, float] = {}
for a in T1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    T1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_t1_robot_cfg())

  viewer.launch(robot.spec.compile())
