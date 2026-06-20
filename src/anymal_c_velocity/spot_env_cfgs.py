"""Boston Dynamics Spot velocity environment configurations."""

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


def illegal_nonfoot_contact(
  env,
  sensor_name: str,
) -> torch.Tensor:
  """Terminate when any monitored non-foot geom contacts terrain."""

  sensor = env.scene[sensor_name]

  if sensor.data.found is None:
    raise RuntimeError(f"Contact sensor '{sensor_name}' does not provide found data.")

  return sensor.data.found.reshape(
    env.num_envs,
    -1,
  ).any(dim=1)


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
    func=illegal_nonfoot_contact,
    params={
      "sensor_name": nonfoot_ground_cfg.name,
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

  # ---------------------------------------------------------------
  # Initial locomotion training conditions
  # ---------------------------------------------------------------

  # Do not apply periodic disturbances while the policy is still
  # discovering basic stepping and command tracking.
  cfg.events.pop("push_robot", None)

  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)

  # Use direct body-frame velocity commands.
  command.heading_command = False
  command.rel_heading_envs = 0.0
  command.ranges.heading = None

  # Do not assign zero commands during the initial gait-discovery run.
  # Every environment must initially attempt forward locomotion.
  command.rel_standing_envs = 0.0

  # Give the policy enough time to establish a gait before resampling.
  command.resampling_time_range = (4.0, 7.0)

  # Initial curriculum stage: nonzero forward walking only.
  command.ranges.lin_vel_x = (0.3, 0.8)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (0.0, 0.0)

  # ---------------------------------------------------------------
  # Command curriculum
  #
  # RSL-RL collects 24 environment steps per learning iteration.
  # Therefore, iteration N corresponds approximately to N * 24
  # environment-control steps.
  # ---------------------------------------------------------------

  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      # Iterations 0–1499:
      # Discover forward stepping.
      "step": 0,
      "lin_vel_x": (0.3, 0.8),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (0.0, 0.0),
    },
    {
      # From approximately iteration 1500:
      # Add backward walking and gentle turning.
      "step": 1500 * 24,
      "lin_vel_x": (-0.5, 1.0),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (-0.25, 0.25),
    },
    {
      # From approximately iteration 3000:
      # Introduce moderate lateral commands.
      "step": 3000 * 24,
      "lin_vel_x": (-0.8, 1.0),
      "lin_vel_y": (-0.35, 0.35),
      "ang_vel_z": (-0.5, 0.5),
    },
    {
      # From approximately iteration 5000:
      # Full omnidirectional target command range.
      "step": 5000 * 24,
      "lin_vel_x": (-1.0, 1.0),
      "lin_vel_y": (-1.0, 1.0),
      "ang_vel_z": (-0.5, 0.5),
    },
  ]

  # ---------------------------------------------------------------
  # Locomotion reward balance
  # ---------------------------------------------------------------

  # Command tracking must be the principal objective.
  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.5

  # Initially this rewards low unwanted rotation. Later it also rewards
  # commanded yaw-rate tracking.
  cfg.rewards["track_angular_velocity"].weight = 1.5

  # Upright posture supports locomotion but does not dominate it.
  cfg.rewards["upright"].weight = 1.0

  # Keep only a weak preference for the home pose. A large pose reward
  # encourages standing still and resists the joint excursions needed
  # for walking.
  cfg.rewards["pose"].weight = 0.15

  cfg.rewards["pose"].params["std_walking"] = {
    ".*_hx": 0.45,
    ".*_hy": 0.45,
    ".*_kn": 0.80,
  }

  cfg.rewards["pose"].params["std_running"] = {
    ".*_hx": 0.55,
    ".*_hy": 0.55,
    ".*_kn": 0.90,
  }

  # Encourage feet to leave the terrain, but keep this reward modest
  # so the policy does not learn excessive hopping.
  cfg.rewards["air_time"].weight = 0.25
  cfg.rewards["air_time"].params["threshold_min"] = 0.08
  cfg.rewards["air_time"].params["threshold_max"] = 0.45
  cfg.rewards["air_time"].params["command_threshold"] = 0.2

  # Desired foot-site height during swing. The standing site height is
  # approximately 0.021 m, so 0.08 m corresponds to about 6 cm of lift.
  cfg.rewards["foot_clearance"].weight = -0.5
  cfg.rewards["foot_clearance"].params["target_height"] = 0.08
  cfg.rewards["foot_clearance"].params["command_threshold"] = 0.2

  cfg.rewards["foot_swing_height"].weight = -0.1
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.08
  cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.2

  # Preserve smoothness and traction penalties without suppressing
  # the exploratory motions needed to discover locomotion.
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["foot_slip"].weight = -0.1
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.2

  cfg.rewards["soft_landing"].weight = -1e-5
  cfg.rewards["soft_landing"].params["command_threshold"] = 0.2

  return cfg
