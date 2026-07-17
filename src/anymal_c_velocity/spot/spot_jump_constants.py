"""Spot configuration used only by jump training."""

from mjlab.entity import EntityCfg

from anymal_c_velocity.spot.spot_constants import (
    SPOT_ARTICULATION_CFG,
    SPOT_COLLISION_CFG,
    SPOT_INIT_STATE,
    get_spec,
)


# The walking task continues to use SPOT_ACTION_SCALE.
# These larger ranges allow the jump policy to reach a deep crouch
# and a nearly extended leg configuration.
SPOT_JUMP_ACTION_SCALE: dict[str, float] = {
    ".*_hx": 0.25,
    ".*_hy": 0.80,
    ".*_kn": 1.25,
}


def get_jump_spec():
    """Load Spot and apply jump torque limits without editing spot.xml."""

    spec = get_spec()

    for actuator in spec.actuators:
        if actuator.name.endswith("_hx"):
            torque_limit = 35.0

        elif actuator.name.endswith(("_hy", "_kn")):
            torque_limit = 200.0

        else:
            continue

        actuator.forcelimited = True
        actuator.forcerange[:] = (
            -torque_limit,
            torque_limit,
        )

    return spec


def get_spot_jump_robot_cfg() -> EntityCfg:
    """Return a separate Spot entity configuration for jump training."""

    return EntityCfg(
        init_state=SPOT_INIT_STATE,
        collisions=(SPOT_COLLISION_CFG,),
        spec_fn=get_jump_spec,
        articulation=SPOT_ARTICULATION_CFG,
    )