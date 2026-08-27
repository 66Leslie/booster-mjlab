"""Booster K1 constants."""

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

_PACKAGED_K1_XML = Path(__file__).parent / "xmls" / "k1" / "robot.xml"


def _resolve_k1_xml() -> Path:
  candidates = (
    Path(os.environ["MJLAB_K1_XML"]) if "MJLAB_K1_XML" in os.environ else None,
    _PACKAGED_K1_XML,
  )
  for path in candidates:
    if path is not None and path.exists():
      return path
  checked = ", ".join(str(path) for path in candidates if path is not None)
  raise FileNotFoundError(
    "K1 MJCF robot.xml was not found. Set MJLAB_K1_XML or sync "
    f"the packaged K1 xmls directory. Checked: {checked}"
  )


def get_spec() -> mujoco.MjSpec:
  xml_path = _resolve_k1_xml()
  xml = xml_path.read_text()
  xml = xml.replace('meshdir="k1/meshes/"', 'meshdir=""')
  meshes_dir = xml_path.parent / "meshes"
  assets = {
    path.name: path.read_bytes() for path in meshes_dir.glob("*") if path.is_file()
  }
  spec = mujoco.MjSpec.from_string(xml, assets=assets)
  while spec.actuators:
    spec.delete(spec.actuators[0])
  return spec


##
# Actuator config.
##


K1_JOINT_NAMES: tuple[str, ...] = (
  "AAHead_yaw",
  "Head_pitch",
  "ALeft_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "ARight_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
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


def _kp(armature: float, natural_freq: float) -> float:
  return armature * (2 * 3.1415926535 * natural_freq) ** 2


def _kv(armature: float, natural_freq: float, damping_ratio: float) -> float:
  return 2.0 * damping_ratio * armature * (2 * 3.1415926535 * natural_freq)


_NF = 4.0

K1_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=("AAHead_yaw", "Head_pitch"),
  stiffness=_kp(0.001, _NF),
  damping=_kv(0.001, _NF, 2.0),
  effort_limit=6.0,
  armature=0.001,
)

K1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
  ),
  stiffness=_kp(0.001, _NF),
  damping=_kv(0.001, _NF, 2.0),
  effort_limit=14.0,
  armature=0.001,
)

K1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Pitch",),
  stiffness=_kp(0.0478125, _NF),
  damping=_kv(0.0478125, _NF, 1.5),
  effort_limit=68.0,
  armature=0.0478125,
)

K1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Roll",),
  stiffness=_kp(0.0339552, _NF),
  damping=_kv(0.0339552, _NF, 1.5),
  effort_limit=76.0,
  armature=0.0339552,
)

K1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Yaw",),
  stiffness=_kp(0.0282528, _NF),
  damping=_kv(0.0282528, _NF, 1.5),
  effort_limit=38.3,
  armature=0.0282528,
)

K1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Knee_Pitch",),
  stiffness=_kp(0.095625, _NF),
  damping=_kv(0.095625, _NF, 1.0),
  effort_limit=112.0,
  armature=0.095625,
)

_ANKLE_ARMATURE = 0.0282528 * 2.0

K1_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Ankle_Pitch",),
  stiffness=_kp(_ANKLE_ARMATURE, _NF),
  damping=_kv(_ANKLE_ARMATURE, _NF, 1.5),
  effort_limit=38.3,
  armature=_ANKLE_ARMATURE,
)

K1_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Ankle_Roll",),
  stiffness=_kp(_ANKLE_ARMATURE, _NF),
  damping=_kv(_ANKLE_ARMATURE, _NF, 1.5),
  effort_limit=38.3,
  armature=_ANKLE_ARMATURE,
)

##
# Keyframes.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.543),
  joint_pos={
    "Left_Shoulder_Roll": -1.3,
    "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Roll": 1.3,
    "Right_Elbow_Yaw": 0.4,
    ".*_Hip_Pitch": -0.2,
    ".*_Knee_Pitch": 0.4,
    ".*_Ankle_Pitch": -0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

K1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    K1_ACTUATOR_HEAD,
    K1_ACTUATOR_ARM,
    K1_ACTUATOR_HIP_PITCH,
    K1_ACTUATOR_HIP_ROLL,
    K1_ACTUATOR_HIP_YAW,
    K1_ACTUATOR_KNEE,
    K1_ACTUATOR_ANKLE_PITCH,
    K1_ACTUATOR_ANKLE_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_k1_robot_cfg() -> EntityCfg:
  """Get a fresh K1 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=K1_ARTICULATION,
  )


K1_ACTION_SCALE: dict[str, float] = {}
for a in K1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  assert a.effort_limit is not None
  for n in a.target_names_expr:
    K1_ACTION_SCALE[n] = 0.25 * a.effort_limit / a.stiffness
