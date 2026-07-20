"""Boston Dynamics Spot velocity environment configurations."""
import os
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

from anymal_c_velocity.spot.ps5_command import Ps5VelocityCommandCfg

@dataclass(kw_only=True)
class SplitVelocityCommandCfg(UniformVelocityCommandCfg):
  """Sample standing, translation, lateral, and pure-yaw commands."""

  # Full forward/backward speed range.
  lin_vel_x_abs_range: tuple[float, float] = (
    0.25,
    1.50,
  )

  forward_probability: float = 0.50

  # Extra low-speed longitudinal examples.
  low_speed_probability: float = 0.35

  low_speed_x_abs_range: tuple[float, float] = (
    0.25,
    0.55,
  )

  # Of the low-speed commands, this fraction is straight:
  # vy = 0 and wz = 0.
  straight_low_speed_probability: float = 0.50

  # Remaining low-speed commands receive only mild diagonal/yaw.
  low_speed_max_abs_y: float = 0.15
  low_speed_max_abs_yaw: float = 0.30

  # Fractions of non-standing commands.
  pure_lateral_probability: float = 0.30
  pure_yaw_probability: float = 0.10

  # Pure-lateral velocity range.
  lin_vel_y_abs_range: tuple[float, float] = (
    0.25,
    0.80,
  )

  left_probability: float = 0.50

  # Pure turning-in-place yaw-rate range.
  ang_vel_z_abs_range: tuple[float, float] = (
    0.20,
    0.80,
  )

  left_turn_probability: float = 0.50

  def __post_init__(self) -> None:
    super().__post_init__()

    min_x, max_x = self.lin_vel_x_abs_range

    if min_x <= 0.0 or max_x < min_x:
      raise ValueError(
        "Invalid lin_vel_x_abs_range."
      )

    low_min_x, low_max_x = (
      self.low_speed_x_abs_range
    )

    if not (
      min_x
      <= low_min_x
      <= low_max_x
      <= max_x
    ):
      raise ValueError(
        "low_speed_x_abs_range must lie inside "
        "lin_vel_x_abs_range."
      )

    min_y, max_y = self.lin_vel_y_abs_range

    if min_y <= 0.0 or max_y < min_y:
      raise ValueError(
        "Invalid lin_vel_y_abs_range."
      )

    min_wz, max_wz = self.ang_vel_z_abs_range

    if min_wz <= 0.0 or max_wz < min_wz:
      raise ValueError(
        "Invalid ang_vel_z_abs_range."
      )

    if self.low_speed_max_abs_y < 0.0:
      raise ValueError(
        "low_speed_max_abs_y cannot be negative."
      )

    if self.low_speed_max_abs_yaw < 0.0:
      raise ValueError(
        "low_speed_max_abs_yaw cannot be negative."
      )

    probabilities = {
      "forward_probability": (
        self.forward_probability
      ),
      "low_speed_probability": (
        self.low_speed_probability
      ),
      "straight_low_speed_probability": (
        self.straight_low_speed_probability
      ),
      "pure_lateral_probability": (
        self.pure_lateral_probability
      ),
      "pure_yaw_probability": (
        self.pure_yaw_probability
      ),
      "left_probability": (
        self.left_probability
      ),
      "left_turn_probability": (
        self.left_turn_probability
      ),
    }

    for name, probability in probabilities.items():
      if not 0.0 <= probability <= 1.0:
        raise ValueError(
          f"{name} must be between 0 and 1."
        )

    if (
      self.pure_lateral_probability
      + self.pure_yaw_probability
      > 1.0
    ):
      raise ValueError(
        "pure_lateral_probability and "
        "pure_yaw_probability cannot sum above 1."
      )

    if self.init_velocity_prob != 0.0:
      raise ValueError(
        "SplitVelocityCommandCfg requires "
        "init_velocity_prob=0.0."
      )

  def build(self, env):
    return SplitVelocityCommand(self, env)


class SplitVelocityCommand(UniformVelocityCommand):
  """Sample standing, translation, lateral, and pure-yaw motion."""

  cfg: SplitVelocityCommandCfg

  def _sample_signed_magnitude(
    self,
    count: int,
    magnitude_range: tuple[float, float],
    positive_probability: float,
  ) -> torch.Tensor:
    """Sample signed values from a nonzero magnitude range."""

    minimum, maximum = magnitude_range

    magnitude = torch.empty(
      count,
      device=self.device,
    ).uniform_(
      minimum,
      maximum,
    )

    positive_mask = (
      torch.rand(
        count,
        device=self.device,
      )
      < positive_probability
    )

    return torch.where(
      positive_mask,
      magnitude,
      -magnitude,
    )

  def _resample_command(
    self,
    env_ids: torch.Tensor,
  ) -> None:
    """Resample commands for the selected environments."""

    # The parent samples the normal vx, vy and yaw commands,
    # chooses standing environments, and updates its internal flags.
    super()._resample_command(env_ids)

    if len(env_ids) == 0:
      return

    # Preserve environments selected by the parent for standing.
    # Their commands remain exactly [0, 0, 0].
    standing_mask = self.is_standing_env[env_ids]
    moving_ids = env_ids[~standing_mask]

    if len(moving_ids) == 0:
      return

    # -----------------------------------------------------------
    # Select command category for every non-standing environment.
    # -----------------------------------------------------------
    category_sample = torch.rand(
      len(moving_ids),
      device=self.device,
    )

    yaw_limit = self.cfg.pure_yaw_probability

    lateral_limit = (
      yaw_limit
      + self.cfg.pure_lateral_probability
    )

    pure_yaw_mask = (
      category_sample < yaw_limit
    )

    pure_lateral_mask = (
      (category_sample >= yaw_limit)
      & (category_sample < lateral_limit)
    )

    longitudinal_mask = ~(
      pure_yaw_mask
      | pure_lateral_mask
    )

    # -----------------------------------------------------------
    # Pure turning in place:
    # vx = 0, vy = 0, wz != 0
    # -----------------------------------------------------------
    pure_yaw_ids = moving_ids[
      pure_yaw_mask
    ]

    if len(pure_yaw_ids) > 0:
      self.vel_command_b[pure_yaw_ids] = 0.0

      yaw_command = self._sample_signed_magnitude(
        len(pure_yaw_ids),
        self.cfg.ang_vel_z_abs_range,
        self.cfg.left_turn_probability,
      )

      self.vel_command_b[
        pure_yaw_ids,
        2,
      ] = yaw_command

    # -----------------------------------------------------------
    # Pure lateral:
    # vx = 0, vy != 0, wz = 0
    # -----------------------------------------------------------
    pure_lateral_ids = moving_ids[
      pure_lateral_mask
    ]

    if len(pure_lateral_ids) > 0:
      self.vel_command_b[pure_lateral_ids] = 0.0

      lateral_command = (
        self._sample_signed_magnitude(
          len(pure_lateral_ids),
          self.cfg.lin_vel_y_abs_range,
          self.cfg.left_probability,
        )
      )

      self.vel_command_b[
        pure_lateral_ids,
        1,
      ] = lateral_command

    # -----------------------------------------------------------
    # Forward/backward and diagonal commands.
    # -----------------------------------------------------------
    longitudinal_ids = moving_ids[
      longitudinal_mask
    ]

    if len(longitudinal_ids) == 0:
      return

    low_speed_mask = (
      torch.rand(
        len(longitudinal_ids),
        device=self.device,
      )
      < self.cfg.low_speed_probability
    )

    low_speed_ids = longitudinal_ids[
      low_speed_mask
    ]

    normal_speed_ids = longitudinal_ids[
      ~low_speed_mask
    ]

    # -----------------------------------------------------------
    # Normal-speed longitudinal commands.
    #
    # Keep the vy and wz sampled by the parent so the policy still
    # trains diagonal movement and translation while turning.
    # -----------------------------------------------------------
    if len(normal_speed_ids) > 0:
      normal_min_x = (
        self.cfg.low_speed_x_abs_range[1]
      )

      normal_max_x = (
        self.cfg.lin_vel_x_abs_range[1]
      )

      x_command = self._sample_signed_magnitude(
        len(normal_speed_ids),
        (
          normal_min_x,
          normal_max_x,
        ),
        self.cfg.forward_probability,
      )

      self.vel_command_b[
        normal_speed_ids,
        0,
      ] = x_command

    # -----------------------------------------------------------
    # Low-speed longitudinal commands.
    # -----------------------------------------------------------
    if len(low_speed_ids) > 0:
      low_x_command = (
        self._sample_signed_magnitude(
          len(low_speed_ids),
          self.cfg.low_speed_x_abs_range,
          self.cfg.forward_probability,
        )
      )

      self.vel_command_b[
        low_speed_ids,
        0,
      ] = low_x_command

      straight_mask = (
        torch.rand(
          len(low_speed_ids),
          device=self.device,
        )
        < self.cfg.straight_low_speed_probability
      )

      straight_ids = low_speed_ids[
        straight_mask
      ]

      mild_combined_ids = low_speed_ids[
        ~straight_mask
      ]

      # Half of the low-speed examples are deliberately simple:
      # slow straight forward or backward movement.
      if len(straight_ids) > 0:
        self.vel_command_b[
          straight_ids,
          1,
        ] = 0.0

        self.vel_command_b[
          straight_ids,
          2,
        ] = 0.0

      # The other half include only mild lateral and yaw commands.
      if len(mild_combined_ids) > 0:
        mild_y = torch.empty(
          len(mild_combined_ids),
          device=self.device,
        ).uniform_(
          -self.cfg.low_speed_max_abs_y,
          self.cfg.low_speed_max_abs_y,
        )

        mild_yaw = torch.empty(
          len(mild_combined_ids),
          device=self.device,
        ).uniform_(
          -self.cfg.low_speed_max_abs_yaw,
          self.cfg.low_speed_max_abs_yaw,
        )

        self.vel_command_b[
          mild_combined_ids,
          1,
        ] = mild_y

        self.vel_command_b[
          mild_combined_ids,
          2,
        ] = mild_yaw


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
  cfg.sim.mujoco.ccd_iterations = 50
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

  if (
    cfg.scene.terrain is not None
    and cfg.scene.terrain.terrain_generator is not None
  ):
    terrain_generator = (
      cfg.scene.terrain.terrain_generator
    )

    terrain_generator.curriculum = True

    sub_terrains = terrain_generator.sub_terrains

    # Limit hills and craters to realistic maximum gradients.
    sub_terrains[
      "hf_pyramid_slope"
    ].slope_range = (0.0, 0.60)

    sub_terrains[
      "hf_pyramid_slope_inv"
    ].slope_range = (0.0, 0.60)

    # Coarsen only the heightfield terrains to reduce
    # heightfield-contact overflow warnings.
    for terrain_name in (
      "hf_pyramid_slope",
      "hf_pyramid_slope_inv",
      "random_rough",
      "wave_terrain",
    ):
      sub_terrains[
        terrain_name
      ].horizontal_scale = 0.20


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

  common_command_kwargs = {
    "entity_name": base_command.entity_name,
    "resampling_time_range": (3.0, 6.0),
    "debug_vis": base_command.debug_vis,
    "heading_command": False,
    "heading_control_stiffness": (
      base_command.heading_control_stiffness
    ),
    "rel_heading_envs": 0.0,
    "init_velocity_prob": 0.0,
    "ranges": UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.5, 1.5),
      lin_vel_y=(-0.4, 0.4),
      ang_vel_z=(-1.0, 1.0),
      heading=None,
    ),
    "viz": UniformVelocityCommandCfg.VizCfg(
      z_offset=0.5,
      scale=base_command.viz.scale,
    ),
  }

  use_ps5_control = (
    play
    and os.environ.get(
      "SPOT_PS5_CONTROL",
      "0",
    ) == "1"
  )

  if use_ps5_control:
    cfg.commands["twist"] = Ps5VelocityCommandCfg(
      **common_command_kwargs,

      # Manual control supplies exact zero when sticks are centered.
      rel_standing_envs=0.0,

      controller_index=0,
      deadzone=0.12,

      min_vx=0.25,
      max_vx=1.50,

      max_diagonal_vy=0.40,

      min_lateral_vy=0.35,
      max_lateral_vy=0.80,

      max_wz= 0.80,
    )

  else:
    cfg.commands["twist"] = SplitVelocityCommandCfg(
      **common_command_kwargs,

      # Five percent of all environments receive an exact
      # zero command, and the custom sampler now preserves them.
      rel_standing_envs=0.05,

      # Full longitudinal range.
      lin_vel_x_abs_range=(0.25, 1.50),
      forward_probability=0.50,

      # More exposure to controlled low-speed motion.
      low_speed_probability=0.35,
      low_speed_x_abs_range=(0.25, 0.55),

      # Half of low-speed commands are straight.
      straight_low_speed_probability=0.50,

      # The other half contain only mild vy and yaw.
      low_speed_max_abs_y=0.15,
      low_speed_max_abs_yaw=0.30,

      # Increase dedicated lateral exposure.
      pure_lateral_probability=0.30,
      lin_vel_y_abs_range=(0.25, 0.80),
      left_probability=0.50,

      # Add explicit turning-in-place training.
      pure_yaw_probability=0.10,
      ang_vel_z_abs_range=(0.20, 0.80),
      left_turn_probability=0.50,
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
    cfg.episode_length_s = int(1e9)

    # Deterministic policy evaluation.
    cfg.observations["actor"].enable_corruption = False

    for event_name in (
      "push_robot",
      "foot_friction",
      "base_com",
      "encoder_bias",
    ):
      cfg.events.pop(event_name, None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        terrain_generator = (
          cfg.scene.terrain.terrain_generator
        )

        terrain_generator.curriculum = False
        terrain_generator.num_cols = 20
        terrain_generator.num_rows = 10
        terrain_generator.border_width = 10.0

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
