"""Spot stand-and-jump environment configuration."""

from dataclasses import replace

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from anymal_c_velocity import spot_jump_mdp as jump_mdp
from anymal_c_velocity.spot.spot_constants import (
    SPOT_ARTICULATION_CFG,
    SPOT_COLLISION_CFG,
    SPOT_INIT_STATE,
    get_spec,
)
from anymal_c_velocity.spot_env_cfgs import spot_flat_env_cfg

JUMP_TASK = jump_mdp.JumpTask(
    height_range=(0.15, 0.2),
    play_height=0.12,
    stand_probability=0.2,

)

NORMAL_JUMP_RESET_PROB = 0.55
RISING_JUMP_RESET_PROB = 0.30

SPOT_JUMP_ACTION_SCALE = {
    ".*_hx": 0.25,
    ".*_hy": 0.80,
    ".*_kn": 1.30,
}


def get_jump_spec() -> mujoco.MjSpec:
    spec = get_spec()
    for joint in spec.joints:
        if joint.name.endswith("_hx"):
            limit = 35.0
        elif joint.name.endswith(("_hy", "_kn")):
            limit = 200.0
        else:
           continue
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.sctfrcrange[:] = (-limit, limit)
    
    return spec


def get_spot_jump_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=SPOT_INIT_STATE,
        collisions=(SPOT_COLLISION_CFG,),
        spec_fn=get_jump_spec,
        articulation=SPOT_ARTICULATION_CFG,
    )


def spot_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = spot_flat_env_cfg(play=False)
    cfg.scene.entities = {"robot": get_spot_jump_robot_cfg()}
    cfg.episode_length_s = 4.0
    cfg.is_finite_horizon = True

    task = JUMP_TASK
    normal_reset_prob = NORMAL_JUMP_RESET_PROB
    rising_reset_prob = RISING_JUMP_RESET_PROB 
    if play:
        task = replace(
            JUMP_TASK,
            height_range=(JUMP_TASK.play_height, JUMP_TASK.play_height),
            stand_probability=0.0,
        )
        normal_reset_prob = 1.0
        rising_reset_prob = 0.0

    
    cfg.commands = {
        jump_mdp.COMMAND_NAME: jump_mdp.JumpCommandCfg(
            resampling_time_range=(1000.0, 1000.0),
            debug_vis=False,
        )
    }

    cfg.curriculum.clear()
    for name in ("push_robot", "foot_friction", "base_com", "encoder_bias"):
        cfg.events.pop
        