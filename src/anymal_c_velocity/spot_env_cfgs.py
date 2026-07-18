"""Boston Dynamics Spot velocity environment configurations."""

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from anymal_c_velocity.spot.spot_constants import (
  SPOT_ACTION_SCALE,
  get_spot_robot_cfg,
)


@dataclass(kw_only=True)
class SplitVelocityCommandCfg(UniformVelocityCommandCfg):
  """Sample longitudinal, diagonal, and pure-lateral commands."""

  # Magnitude of vx for forward/backward-moving environments.
  lin_vel_x_abs_range: tuple[float, float] = (0.35, 0.65)

  # Probability that a longitudinal command points forward.
  forward_probability: float = 0.5

  # Fraction of moving commands that will be purely lateral.
  pure_lateral_probability: float = 0.20

  # Magnitude of vy for pure-lateral environments.
  lin_vel_y_abs_range: tuple[float, float] = (0.20, 0.40)

  # Probability of positive body-frame y motion.
  # For your Spot convention, positive y is left.
  left_probability: float = 0.5

  def __post_init__(self) -> None:
    super().__post_init__()

    min_x, max_x = self.lin_vel_x_abs_range
    if min_x <= 0.0:
      raise ValueError(
        "lin_vel_x_abs_range minimum must be greater than zero."
      )
    if max_x < min_x:
      raise ValueError(
        "lin_vel_x_abs_range maximum must be greater than "
        "or equal to its minimum."
      )

    min_y, max_y = self.lin_vel_y_abs_range
    if min_y <= 0.0:
      raise ValueError(
        "lin_vel_y_abs_range minimum must be greater than zero."
      )
    if max_y < min_y:
      raise ValueError(
        "lin_vel_y_abs_range maximum must be greater than "
        "or equal to its minimum."
      )

    if not 0.0 <= self.forward_probability <= 1.0:
      raise ValueError(
        "forward_probability must be between 0 and 1."
      )

    if not 0.0 <= self.pure_lateral_probability <= 1.0:
      raise ValueError(
        "pure_lateral_probability must be between 0 and 1."
      )

    if not 0.0 <= self.left_probability <= 1.0:
      raise ValueError(
        "left_probability must be between 0 and 1."
      )

    if self.init_velocity_prob != 0.0:
      raise ValueError(
        "SplitVelocityCommandCfg currently requires "
        "init_velocity_prob=0.0."
      )

  def build(self, env):
    return SplitVelocityCommand(self, env)


class SplitVelocityCommand(UniformVelocityCommand):
  """Sample longitudinal/diagonal or pure-lateral motion."""

  cfg: SplitVelocityCommandCfg

  def _resample_command(
    self,
    env_ids: torch.Tensor,
  ) -> None:
    # Standard MjLab sampling handles:
    # - vy for ordinary longitudinal/diagonal commands
    # - yaw velocity
    # - standing environments
    # - heading environments
    super()._resample_command(env_ids)

    num_envs = len(env_ids)
    if num_envs == 0:
      return

    # Select which environments receive pure-lateral commands.
    lateral_mask = (
      torch.rand(
        num_envs,
        device=self.device,
      )
      < self.cfg.pure_lateral_probability
    )

    longitudinal_mask = ~lateral_mask

    # -------------------------------------------------------------
    # Forward/backward or diagonal commands
    # -------------------------------------------------------------
    longitudinal_ids = env_ids[longitudinal_mask]

    if len(longitudinal_ids) > 0:
      min_x, max_x = self.cfg.lin_vel_x_abs_range

      x_magnitude = torch.empty(
        len(longitudinal_ids),
        device=self.device,
      ).uniform_(min_x, max_x)

      forward_mask = (
        torch.rand(
          len(longitudinal_ids),
          device=self.device,
        )
        < self.cfg.forward_probability
      )

      x_direction = torch.where(
        forward_mask,
        torch.ones_like(x_magnitude),
        -torch.ones_like(x_magnitude),
      )

      self.vel_command_b[longitudinal_ids, 0] = (
        x_direction * x_magnitude
      )

      # vy is left unchanged here. It retains the value sampled from:
      # ranges.lin_vel_y
      #
      # Therefore:
      # ranges.lin_vel_y = (-0.15, 0.15)
      # produces mild diagonal commands.

    # -------------------------------------------------------------
    # Pure sideways commands
    # -------------------------------------------------------------
    lateral_ids = env_ids[lateral_mask]

    if len(lateral_ids) > 0:
      min_y, max_y = self.cfg.lin_vel_y_abs_range

      y_magnitude = torch.empty(
        len(lateral_ids),
        device=self.device,
      ).uniform_(min_y, max_y)

      left_mask = (
        torch.rand(
          len(lateral_ids),
          device=self.device,
        )
        < self.cfg.left_probability
      )

      y_direction = torch.where(
        left_mask,
        torch.ones_like(y_magnitude),
        -torch.ones_like(y_magnitude),
      )

      # Pure lateral means no longitudinal motion.
      self.vel_command_b[lateral_ids, 0] = 0.0
      self.vel_command_b[lateral_ids, 1] = (
        y_direction * y_magnitude
      )


def terrain_contact_detected(
  env,
  sensor_name: str,
) -> torch.Tensor:
  """Return whether selected robot geoms contact the terrain."""

  sensor = env.scene[sensor_name]

  if sensor.data.found is None:
    raise RuntimeError(f"Contact sensor '{sensor_name}' does not provide found data.")

  return sensor.data.found.reshape(
    env.num_envs,
    -1,
  ).any(dim=1)


def excessive_foot_air_time(
  env,
  sensor_name: str,
  command_name: str,
  command_threshold: float,
  max_air_time: float,
) -> torch.Tensor:
  """Penalize feet that remain continously airborne too long."""

  sensor = env.scene[sensor_name]

  if sensor.data.current_air_time is None:
    raise RuntimeError(f"Sensor '{sensor_name}' must use track_air_time=True.")

  air_time = sensor.data.current_air_time

  # Zero below the permitted air time.
  # Scale and clamp the penalty to keep it bounded in [0, 1]
  excess = torch.clamp(
    (air_time - max_air_time) / max_air_time,
    min=0.0,
    max=1.0,
  )

  penalty = torch.max(excess, dim=1).values

  command = env.command_manager.get_command(command_name)

  command_magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(
    command[:, 2]
  )

  return penalty * (command_magnitude > command_threshold).float()


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
  cfg.sim.nconmax = 128

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
  # nonfoot_ground_cfg = ContactSensorCfg(
  #   name="nonfoot_ground_touch",
  #   primary=ContactMatch(
  #     mode="geom",
  #     entity="robot",
  #     pattern=r".*_collision$",
  #     # pattern="body_collision",
  #     exclude=geom_names,
  #   ),
  #   secondary=ContactMatch(
  #     mode="body",
  #     pattern="terrain",
  #   ),
  #   fields=("found",),
  #   reduce="none",
  #   num_slots=1,
  # )

  # Contacts that indicate an actual fall or sever collision
  terminal_ground_cfg = ContactSensorCfg(
    name="terminal_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r"(body_collision|.*_uleg_collision)",
    ),
    secondary=ContactMatch(
      mode="body",
      pattern="terrain",
    ),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  # Lower-leg contacts are undesirable but recoverable on stairs
  lower_leg_ground_cfg = ContactSensorCfg(
    name="lower_leg_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_lleg_collision",
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
    terminal_ground_cfg,
    lower_leg_ground_cfg,
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
    func=terrain_contact_detected,
    params={
      "sensor_name": terminal_ground_cfg.name,
    },
  )

  # Use the same omnidirectional command distribution on every terrain
  base_command = cfg.commands["twist"]
  assert isinstance(
    base_command,
    UniformVelocityCommandCfg,
  )

  cfg.commands["twist"] = SplitVelocityCommandCfg(
    entity_name=base_command.entity_name,

    resampling_time_range=(3.0, 6.0),
    debug_vis=base_command.debug_vis,

    heading_command=False,
    heading_control_stiffness=(
      base_command.heading_control_stiffness
    ),
    rel_heading_envs=0.0,

    # Exact standing commands.
    rel_standing_envs=0.10,

    init_velocity_prob=0.0,

    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.5, 1.5),
      lin_vel_y=(-0.4, 0.4),
      ang_vel_z=(-1.0, 1.0),
      heading=None,
    ),

    viz=UniformVelocityCommandCfg.VizCfg(
      z_offset=0.5,
      scale=base_command.viz.scale,
    ),

    lin_vel_x_abs_range=(0.15, 1.5),
    forward_probability=0.50,

    pure_lateral_probability=0.20,

    lin_vel_y_abs_range=(0.15, 0.80),
    left_probability=0.50,
  )

  
  # Prevent the inherited curriculum from chaning this distribution.
  cfg.curriculum.pop("command_vel", None)

  # Shared locomotion reward balance
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.5

  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["upright"].weight = 1.0
  cfg.rewards["pose"].weight = 1.0

  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["excessive_air_time"] = RewardTermCfg(
    func=excessive_foot_air_time,
    weight=-0.35,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.2,
      "max_air_time": 0.35,
    },
  )

  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards["foot_slip"].weight = -0.1
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.2

  cfg.rewards["soft_landing"].weight = -1e-5
  cfg.rewards["soft_landing"].params["command_threshold"] = 0.2

  cfg.rewards["foot_clearance"].weight = 0.0
  cfg.rewards["foot_swing_height"].weight = 0.0

  cfg.rewards["lower_leg_contact"] = RewardTermCfg(
    func=terrain_contact_detected,
    weight=-0.5,
    params={
      "sensor_name": lower_leg_ground_cfg.name,
    },
  )


  if play:
    # Allow continuous viewing without short episode timeouts.
    cfg.episode_length_s = int(1e9)

    # Disable observation noise and external pushes while debugging.
    cfg.observations["actor"].enable_corruption = False

    # cfg.events.pop("push_robot", None)
    push_event = cfg.events["push_robot"]
    push_event.interval_range_s = (2.0, 5.0)

    push_event.params["velocity_range"] = {
      "x": (-0.25, 0.25),
      "y": (-0.5, 0.5),
      "z": (0.0, 0.0),
      "roll": (0.0, 0.0),
      "pitch": (0.0, 0.0),
      "yaw": (-0.15, 0.15),
    }

    # cfg.terminations.pop("illegal_contact", None)
    # Deterministic policy evaluation.
    for event_name in (
      "push_robot",
      "foot_friction",
      "base_com",
      "encoder_bias",
    ):
      cfg.events.pop(event_name, None)

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

  # ---------------------------------------------------------------
  # Locomotion reward balance
  # ---------------------------------------------------------------

  # Stronger swing-foot trajectory shaping.
  cfg.rewards["foot_clearance"].weight = -2.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.08
  cfg.rewards["foot_clearance"].params["command_threshold"] = 0.2

  cfg.rewards["foot_swing_height"].weight = -0.25
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.08
  cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.2


  return cfg
