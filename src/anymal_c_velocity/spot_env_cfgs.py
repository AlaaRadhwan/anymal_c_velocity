"""Boston Dynamics Spot velocity environment configurations."""

import math

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from anymal_c_velocity.spot.spot_constants import (
  SPOT_ACTION_SCALE,
  get_spot_robot_cfg,
)


def illegal_contact_after_grace(
  env,
  sensor_name: str,
  grace_period_s: float,
) -> torch.Tensor:
  """Terminate on any non-foot contact after reset settling."""

  sensor = env.scene[sensor_name]

  if sensor.data.found is None:
    raise RuntimeError(f"Contact sensor '{sensor_name}' does not provide found data.")

  # Collapse every sensor/contact dimension into one Boolean
  # result for each parallel environment.
  contact_detected = sensor.data.found.reshape(
    env.num_envs,
    -1,
  ).any(dim=1)

  grace_steps = math.ceil(grace_period_s / env.step_dt)

  grace_finished = env.episode_length_buf >= grace_steps

  return contact_detected & grace_finished


def spot_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Spot rough-terrain velocity environment."""

  # Start with MjLab's generic velocity-tracking task.
  cfg = make_velocity_env_cfg()

  # Restore the solver configuration used by the original Spot model.
  cfg.sim.mujoco.timestep = 0.002
  cfg.sim.mujoco.integrator = "implicitfast"
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.mujoco.impratio = 100.0

  cfg.sim.mujoco.solver = "newton"
  cfg.sim.mujoco.iterations = 100
  cfg.sim.mujoco.ls_iterations = 50

  # Preserve a 50 Hz policy/control frequency:
  # 0.002 s × 10 physics steps = 0.020 s per action.
  cfg.decimation = 10

  # Increase simulation/contact capacities for the quadruped.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 50

  # Replace the generic robot with our Spot entity.
  cfg.scene.entities = {
    "robot": get_spot_robot_cfg(),
  }

  # The terrain scanner must follow Spot's base body.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "body"

  # Named reference sites and physical foot geoms.
  site_names = (
    "FL_site",
    "FR_site",
    "HL_site",
    "HR_site",
  )

  geom_names = (
    "FL",
    "FR",
    "HL",
    "HR",
  )

  # Detect contact between each foot and the terrain.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=geom_names,
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="body",
      pattern="terrain",
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  # Detect body or leg contact with the terrain.
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision$",
      # pattern="body_collision",
      exclude=geom_names,
    ),
    secondary=ContactMatch(
      mode="body",
      pattern="terrain",
    ),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  # Enable terrain difficulty progression for rough-terrain training.
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Convert policy outputs into joint-position offsets.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = SPOT_ACTION_SCALE

  # Viewer follows Spot's main body.
  cfg.viewer.body_name = "body"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0

  # Tell observation terms which sites represent the feet.
  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  # Robot-specific domain-randomization targets.
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  cfg.events["base_com"].params["asset_cfg"].body_names = ("body",)

  # Acceptable deviations from the nominal standing pose.
  cfg.rewards["pose"].params["std_standing"] = {
    ".*_hx": 0.05,
    ".*_hy": 0.05,
    ".*_kn": 0.10,
  }

  cfg.rewards["pose"].params["std_walking"] = {
    ".*_hx": 0.30,
    ".*_hy": 0.30,
    ".*_kn": 0.60,
  }

  cfg.rewards["pose"].params["std_running"] = {
    ".*_hx": 0.30,
    ".*_hy": 0.30,
    ".*_kn": 0.60,
  }

  # Robot-specific base body used by reward terms.
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("body",)

  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("body",)

  # Robot-specific feet used by locomotion rewards.
  for reward_name in (
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
  ):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Leave these disabled initially.
  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  # End the episode when a non-foot collision geom touches the floor.
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=illegal_contact_after_grace,
    params={
      "sensor_name": nonfoot_ground_cfg.name,
      "grace_period_s": 0.6,
    },
  )

  # Raise command visualization above the robot.
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.viz.z_offset = 0.5

  if play:
    # Allow continuous viewing without short episode timeouts.
    cfg.episode_length_s = int(1e9)

    # Disable observation noise and external pushes while debugging.
    cfg.observations["actor"].enable_corruption = False

    cfg.events.pop("push_robot", None)
    # cfg.terminations.pop("illegal_contact", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def spot_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Spot flat-terrain velocity environment."""

  cfg = spot_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  # Replace generated rough terrain with a flat plane.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Flat terrain does not require a height scanner.
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )

  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # No terrain-level curriculum is needed on a plane.
  cfg.curriculum.pop("terrain_levels", None)

  return cfg
