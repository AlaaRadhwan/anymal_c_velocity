"""Boston Dynamics Spot configuration for MjLab"""

from pathlib import Path

import mujoco
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

# MJCF model and assets

_HERE = Path(__file__).parent
SPOT_XML = _HERE / "xmls" / "spot.xml"

if not SPOT_XML.exists():
  raise FileNotFoundError(f"Spot MJCF file not found at {SPOT_XML}")


def get_assets(meshdir: str) -> dict[str, bytes]:
  """Load mesh assets referenced by the spot MJCF model."""

  assets: dict[str, bytes] = {}

  update_assets(
    assets,
    SPOT_XML.parent / "assets",
    meshdir,
  )

  return assets


def get_spec() -> mujoco.MjSpec:
  """Load a fresh MuJoCo specification for Spot."""

  spec = mujoco.MjSpec.from_file(str(SPOT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


# Actuators

SPOT_ACTUATOR_CFG = XmlPositionActuatorCfg(
  target_names_expr=(
    ".*_hx",
    ".*_hy",
    ".*_kn",
  ),
)

# Initial standing state
SPOT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.46),
  joint_pos={
    ".*_hx": 0.0,
    ".*_hy": 1.04,
    ".*_kn": -1.80,
  },
  joint_vel={
    ".*": 0.0,
  },
)

# Collision configuration

FOOT_GEOM_REGEX = r"^(FL|FR|HL|HR)$"

SPOT_COLLISION_CFG = CollisionCfg(
    geom_names_expr=(
        r".*_collision$",
        FOOT_GEOM_REGEX,
    ),
    condim={
        r".*_collision$": 3,
        FOOT_GEOM_REGEX: 6,
    },
    priority={
        r".*_collision$": 0,
        FOOT_GEOM_REGEX: 1,
    },
    friction={
        r".*_collision$": (0.6,),
        FOOT_GEOM_REGEX: (0.8, 0.02, 0.01),
    },
    solimp={
        FOOT_GEOM_REGEX: (0.015, 1.0, 0.036),
    },
)


# Articulation configuration

SPOT_ARTICULATION_CFG = EntityArticulationInfoCfg(
  actuators=(SPOT_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)


def get_spot_robot_cfg() -> EntityCfg:
  """Return a fresh Spot entity configuration."""

  return EntityCfg(
    init_state=SPOT_INIT_STATE,
    collisions=(SPOT_COLLISION_CFG,),
    spec_fn=get_spec,
    articulation=SPOT_ARTICULATION_CFG,
  )


# Conservative initial action ranges, measured in radians.
#
# The neural-network output will later be multiplied by these values and
# added to the nominal standing joint position.
SPOT_ACTION_SCALE: dict[str, float] = {
  ".*_hx": 0.20,
  ".*_hy": 0.25,
  ".*_kn": 0.30,
}
