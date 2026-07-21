"""Direct native-MuJoCo backend for the exported Spot policy."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from ..config import PolicyConfig
from ..robot_types import JointTargets, RobotState
from .base import RobotBackend


class MujocoBackend(RobotBackend):
  """Run Spot directly in native MuJoCo without MjLab or RSL-RL."""

  def __init__(
    self,
    xml_path: str | Path,
    config: PolicyConfig,
  ) -> None:
    self._xml_path = Path(xml_path).expanduser().resolve()
    if not self._xml_path.is_file():
      raise FileNotFoundError(f"Spot MJCF file not found: {self._xml_path}")

    self._config = config
    self._model = self._build_model(self._xml_path, config)
    self._data = mujoco.MjData(self._model)

    self._body_id = self._require_id(
      mujoco.mjtObj.mjOBJ_BODY,
      "body",
    )
    self._home_key_id = self._require_id(
      mujoco.mjtObj.mjOBJ_KEY,
      "home",
    )

    self._joint_ids = np.asarray(
      [
        self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in config.joint_order
      ],
      dtype=np.int32,
    )
    self._actuator_ids = np.asarray(
      [
        self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in config.joint_order
      ],
      dtype=np.int32,
    )

    self._validate_joint_actuator_mapping()

    self._qpos_addresses = self._model.jnt_qposadr[self._joint_ids].copy()
    self._dof_addresses = self._model.jnt_dofadr[self._joint_ids].copy()

    self._lin_vel_sensor = self._sensor_slice("imu_lin_vel", expected_dim=3)
    self._ang_vel_sensor = self._sensor_slice("imu_ang_vel", expected_dim=3)

    self._default_targets = np.asarray(
      config.default_joint_positions,
      dtype=np.float64,
    )
    self.reset()

  @property
  def model(self) -> mujoco.MjModel:
    """Expose the compiled model for the native MuJoCo viewer."""

    return self._model

  @property
  def data(self) -> mujoco.MjData:
    """Expose simulation data for the native MuJoCo viewer."""

    return self._data

  @property
  def timestep_s(self) -> float:
    return float(self._model.opt.timestep)

  @property
  def time_s(self) -> float:
    return float(self._data.time)

  @property
  def base_position_world(self) -> NDArray[np.float32]:
    """Return the base-link position in world coordinates."""

    return np.asarray(
      self._data.xpos[self._body_id],
      dtype=np.float32,
    ).copy()
  
  @property
  def base_body_id(self) -> int:
    """Return Spot's base-body ID for viewer tracking."""

    return self._body_id

  def reset(self) -> None:
    """Reset to the MJCF ``home`` keyframe and nominal controls."""

    mujoco.mj_resetDataKeyframe(
      self._model,
      self._data,
      self._home_key_id,
    )
    self._data.ctrl[self._actuator_ids] = self._default_targets
    mujoco.mj_forward(self._model, self._data)

  def read_state(self) -> RobotState:
    """Read one policy state using the same frame conventions as training."""

    base_lin_vel = self._read_sensor(self._lin_vel_sensor)
    base_ang_vel = self._read_sensor(self._ang_vel_sensor)

    rotation_body_to_world = self._data.xmat[self._body_id].reshape(3, 3)
    gravity_world = np.array((0.0, 0.0, -1.0), dtype=np.float64)
    projected_gravity = rotation_body_to_world.T @ gravity_world

    joint_pos = self._data.qpos[self._qpos_addresses]
    joint_vel = self._data.qvel[self._dof_addresses]

    return RobotState(
      base_lin_vel=np.asarray(base_lin_vel, dtype=np.float32),
      base_ang_vel=np.asarray(base_ang_vel, dtype=np.float32),
      projected_gravity=np.asarray(projected_gravity, dtype=np.float32),
      joint_pos=np.asarray(joint_pos, dtype=np.float32),
      joint_vel=np.asarray(joint_vel, dtype=np.float32),
    )

  def write_joint_targets(self, targets: JointTargets) -> None:
    """Write absolute position targets to the matching XML actuators."""

    if targets.positions.size != self._config.action_size:
      raise ValueError(
        f"Expected {self._config.action_size} joint targets; "
        f"got {targets.positions.size}."
      )

    self._data.ctrl[self._actuator_ids] = targets.positions

  def step(self) -> None:
    """Advance native MuJoCo by one configured physics step."""

    mujoco.mj_step(self._model, self._data)

  @staticmethod
  def _build_model(
    xml_path: Path,
    config: PolicyConfig,
  ) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(xml_path))

    # Preserve mesh loading after editing and recompiling the MjSpec.
    assets: dict[str, bytes] = {}
    if spec.meshdir:
      asset_root = xml_path.parent / spec.meshdir
      if not asset_root.is_dir():
        raise FileNotFoundError(
          f"MJCF mesh directory not found: {asset_root}"
        )

      mesh_prefix = Path(spec.meshdir)
      for asset_path in asset_root.rglob("*"):
        if asset_path.is_file():
          relative_path = asset_path.relative_to(asset_root)
          asset_key = (mesh_prefix / relative_path).as_posix()
          assets[asset_key] = asset_path.read_bytes()

    spec.assets = assets

    # Reproduce SPOT_COLLISION_CFG without importing the training package.
    foot_names = {"FL", "FR", "HL", "HR"}

    for geom in spec.geoms:
      # Scene geometry must retain normal collision settings.
      if geom.name == "terrain":
        continue

      if geom.name in foot_names:
        geom.condim = 6
        geom.priority = 1

        for index, value in enumerate((0.8, 0.02, 0.01)):
          geom.friction[index] = value

        for index, value in enumerate(
          (0.015, 1.0, 0.036, 0.5, 2.0)
        ):
          geom.solimp[index] = value

      elif geom.name.endswith("_collision"):
        geom.condim = 3
        geom.priority = 0
        geom.friction[0] = 0.6

      else:
        # Visual-only robot geometries.
        geom.contype = 0
        geom.conaffinity = 0

    # The robot MJCF contains only the robot. MjLab added this same flat plane
    # as the terrain during training and playback.
    # terrain_body = spec.worldbody.add_body(name="terrain")
    # terrain_body.add_geom(
    #   name="terrain",
    #   type=mujoco.mjtGeom.mjGEOM_PLANE,
    #   size=(0.0, 0.0, 0.01),
    # )

    model = spec.compile()

    # Match spot_flat_env_cfg() and MjLab's default solver tolerances.
    model.opt.timestep = config.physics_timestep_s
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = 100.0
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 100
    model.opt.tolerance = 1e-8
    model.opt.ls_iterations = 50
    model.opt.ls_tolerance = 0.01
    model.opt.ccd_iterations = 50
    model.opt.gravity[:] = (0.0, 0.0, -9.81)

    return model

  def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(self._model, object_type, name)
    if object_id < 0:
      raise ValueError(
        f"Required MuJoCo object {name!r} of type {object_type.name} "
        "was not found in the model."
      )
    return int(object_id)

  def _sensor_slice(self, name: str, expected_dim: int) -> slice:
    sensor_id = self._require_id(mujoco.mjtObj.mjOBJ_SENSOR, name)
    sensor_dim = int(self._model.sensor_dim[sensor_id])
    if sensor_dim != expected_dim:
      raise ValueError(
        f"Sensor {name!r} has dimension {sensor_dim}; "
        f"expected {expected_dim}."
      )

    start = int(self._model.sensor_adr[sensor_id])
    return slice(start, start + sensor_dim)

  def _read_sensor(self, sensor_slice: slice) -> NDArray[np.float64]:
    values = np.asarray(self._data.sensordata[sensor_slice], dtype=np.float64)
    if not np.all(np.isfinite(values)):
      raise RuntimeError("MuJoCo sensor returned NaN or infinite values.")
    return values.copy()

  def _validate_joint_actuator_mapping(self) -> None:
    for joint_name, joint_id, actuator_id in zip(
      self._config.joint_order,
      self._joint_ids,
      self._actuator_ids,
      strict=True,
    ):
      joint_type = self._model.jnt_type[joint_id]
      if joint_type != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError(
          f"Joint {joint_name!r} must be a scalar hinge joint."
        )

      transmitted_joint_id = int(self._model.actuator_trnid[actuator_id, 0])
      if transmitted_joint_id != int(joint_id):
        raise ValueError(
          f"Actuator {joint_name!r} does not drive the matching joint."
        )
