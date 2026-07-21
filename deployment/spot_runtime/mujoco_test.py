"""Run the exported Spot policy directly in native MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco.viewer

from .action_processor import ActionProcessor
from .backends.mujoco_backend import MujocoBackend
from .config import PolicyConfig
from .observation_builder import ObservationBuilder
from .policy_runtime import TorchScriptPolicy
from .robot_types import VelocityCommand


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_POLICY = (
  _PROJECT_ROOT / "artifacts" / "policies" / "spot_velocity" / "policy_jit.pt"
)
_DEFAULT_METADATA = (
  _PROJECT_ROOT
  / "artifacts"
  / "policies"
  / "spot_velocity"
  / "policy_metadata.json"
)
_DEFAULT_XML = (
  _PROJECT_ROOT
  / "src"
  / "anymal_c_velocity"
  / "spot"
  / "xmls"
  / "scene.xml"
)

_COMMAND_LIMITS = {
  "vx": 0.65,
  "vy": 0.40,
  "wz": 0.50,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run the exported Spot policy in direct native MuJoCo."
  )
  parser.add_argument("--policy", type=Path, default=_DEFAULT_POLICY)
  parser.add_argument("--metadata", type=Path, default=_DEFAULT_METADATA)
  parser.add_argument("--xml", type=Path, default=_DEFAULT_XML)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--vx", type=float, default=0.0, help="Forward speed in m/s.")
  parser.add_argument("--vy", type=float, default=0.0, help="Leftward speed in m/s.")
  parser.add_argument("--wz", type=float, default=0.0, help="Yaw rate in rad/s.")
  parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Run time in seconds. Zero means until the viewer closes.",
  )
  parser.add_argument(
    "--headless",
    action="store_true",
    help="Run without opening the native MuJoCo viewer.",
  )
  return parser.parse_args()


def validate_command(command: VelocityCommand) -> None:
  values = {
    "vx": command.vx,
    "vy": command.vy,
    "wz": command.wz,
  }
  for name, value in values.items():
    limit = _COMMAND_LIMITS[name]
    if abs(value) > limit:
      raise ValueError(
        f"{name}={value} exceeds the trained absolute limit of {limit}."
      )


def run_loop(
  backend: MujocoBackend,
  config: PolicyConfig,
  policy: TorchScriptPolicy,
  observation_builder: ObservationBuilder,
  action_processor: ActionProcessor,
  command: VelocityCommand,
  viewer_handle,
  duration_s: float,
) -> None:
  start_wall_time = time.perf_counter()
  next_cycle_time = start_wall_time
  next_status_time = start_wall_time

  while viewer_handle is None or viewer_handle.is_running():
    now = time.perf_counter()
    if duration_s > 0.0 and now - start_wall_time >= duration_s:
      break

    state = backend.read_state()
    observation = observation_builder.build(state, command)
    action = policy.act(observation)
    targets = action_processor.process(action)

    backend.write_joint_targets(targets)
    observation_builder.update_action(action)

    for _ in range(config.control_decimation):
      backend.step()

    if viewer_handle is not None:
      viewer_handle.sync()

    now = time.perf_counter()
    if now >= next_status_time:
      base_position = backend.base_position_world
      print(
        f"sim={backend.time_s:7.2f} s  "
        f"base_z={base_position[2]:.3f} m  "
        f"v_body=({state.base_lin_vel[0]:+.3f}, "
        f"{state.base_lin_vel[1]:+.3f}) m/s  "
        f"yaw_rate={state.base_ang_vel[2]:+.3f} rad/s"
      )
      next_status_time = now + 1.0

    next_cycle_time += config.policy_period_s
    sleep_time = next_cycle_time - time.perf_counter()
    if sleep_time > 0.0:
      time.sleep(sleep_time)
    else:
      # Do not accumulate an ever-growing delay after a slow frame.
      next_cycle_time = time.perf_counter()


def main() -> None:
  args = parse_args()

  command = VelocityCommand(vx=args.vx, vy=args.vy, wz=args.wz)
  validate_command(command)

  config = PolicyConfig.from_json(args.metadata)
  policy = TorchScriptPolicy(args.policy, config, device=args.device)
  observation_builder = ObservationBuilder(config)
  action_processor = ActionProcessor(config)
  backend = MujocoBackend(args.xml, config)

  observation_builder.reset()
  backend.reset()

  duration_s = args.duration
  if args.headless and duration_s <= 0.0:
    duration_s = 5.0

  print("Direct MuJoCo Spot test started.")
  print(
    f"  command: vx={command.vx:+.3f} m/s, "
    f"vy={command.vy:+.3f} m/s, wz={command.wz:+.3f} rad/s"
  )
  print(
    f"  timing:  {config.physics_timestep_s:.4f} s x "
    f"{config.control_decimation} = {config.policy_period_s:.3f} s"
  )

  try:
    if args.headless:
      run_loop(
        backend,
        config,
        policy,
        observation_builder,
        action_processor,
        command,
        viewer_handle=None,
        duration_s=duration_s,
      )
    else:
      with mujoco.viewer.launch_passive(
        backend.model,
        backend.data,
        show_left_ui=False,
        show_right_ui=False,
      ) as viewer_handle:
        viewer_handle.cam.lookat[:] = (0.0, 0.0, 0.35)
        viewer_handle.cam.distance = 2.0
        viewer_handle.cam.azimuth = 135.0
        viewer_handle.cam.elevation = -10.0

        run_loop(
          backend,
          config,
          policy,
          observation_builder,
          action_processor,
          command,
          viewer_handle=viewer_handle,
          duration_s=duration_s,
        )
  finally:
    backend.close()

  print("Direct MuJoCo Spot test finished.")


if __name__ == "__main__":
  main()
