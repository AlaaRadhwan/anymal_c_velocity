"""Control the exported Spot policy with a PS5 controller in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer

from .action_processor import ActionProcessor
from .backends.mujoco_backend import MujocoBackend
from .config import PolicyConfig
from .gamepad import (
  Gamepad,
  GamepadDisconnectedError,
  GamepadMapping,
  SpotGamepad,
)
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


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Control Spot in native MuJoCo using a PS5 controller."
  )
  parser.add_argument("--policy", type=Path, default=_DEFAULT_POLICY)
  parser.add_argument("--metadata", type=Path, default=_DEFAULT_METADATA)
  parser.add_argument("--xml", type=Path, default=_DEFAULT_XML)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--gamepad-index", type=int, default=0)
  parser.add_argument("--deadzone", type=float, default=0.10)

  parser.add_argument("--max-vx", type=float, default=0.65)
  parser.add_argument("--max-vy", type=float, default=0.40)
  parser.add_argument("--max-wz", type=float, default=0.50)

  parser.add_argument("--left-x-axis", type=int, default=0)
  parser.add_argument("--left-y-axis", type=int, default=1)
  parser.add_argument("--right-x-axis", type=int, default=3)
  parser.add_argument("--enable-button", type=int, default=4)
  parser.add_argument("--reset-button", type=int, default=0)
  parser.add_argument("--exit-button", type=int, default=1)
  return parser.parse_args()


def run_policy_cycle(
  *,
  backend: MujocoBackend,
  config: PolicyConfig,
  policy: TorchScriptPolicy,
  observation_builder: ObservationBuilder,
  action_processor: ActionProcessor,
  command: VelocityCommand,
) -> None:
  """Evaluate the actor once and hold its targets for one control period."""

  state = backend.read_state()
  observation = observation_builder.build(state, command)
  action = policy.act(observation)
  targets = action_processor.process(action)

  backend.write_joint_targets(targets)
  observation_builder.update_action(action)

  for _ in range(config.control_decimation):
    backend.step()


def main() -> None:
  args = parse_args()

  config = PolicyConfig.from_json(args.metadata)
  policy = TorchScriptPolicy(args.policy, config, device=args.device)
  observation_builder = ObservationBuilder(config)
  action_processor = ActionProcessor(config)
  backend = MujocoBackend(args.xml, config)

  mapping = GamepadMapping(
    left_x_axis=args.left_x_axis,
    left_y_axis=args.left_y_axis,
    right_x_axis=args.right_x_axis,
    enable_button=args.enable_button,
    reset_button=args.reset_button,
    exit_button=args.exit_button,
  )

  observation_builder.reset()
  backend.reset()

  try:
    with Gamepad(device_index=args.gamepad_index) as gamepad:
      controller = SpotGamepad(
        gamepad,
        mapping=mapping,
        deadzone=args.deadzone,
        max_vx=args.max_vx,
        max_vy=args.max_vy,
        max_wz=args.max_wz,
      )

      print("Spot PS5 teleoperation started.")
      print(f"  controller: {gamepad.info.name}")
      print(
        f"  limits:     vx=±{args.max_vx:.2f} m/s, "
        f"vy=±{args.max_vy:.2f} m/s, wz=±{args.max_wz:.2f} rad/s"
      )
      print(
        "  controls:   hold L1 to move, Cross/X to reset, "
        "Circle to exit"
      )
      print(
        "  axes:       left up=+vx, left left=+vy, "
        "right left=+wz"
      )

      with mujoco.viewer.launch_passive(
        backend.model,
        backend.data,
        show_left_ui=False,
        show_right_ui=False,
      ) as viewer_handle:
        with viewer_handle.lock():
          viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
          viewer_handle.cam.trackbodyid = backend.base_body_id
          viewer_handle.cam.distance = 2.0
          viewer_handle.cam.azimuth = 135.0
          viewer_handle.cam.elevation = -10.0

        next_cycle_time = time.perf_counter()
        next_status_time = next_cycle_time

        while viewer_handle.is_running():
          try:
            control = controller.poll()
          except GamepadDisconnectedError as error:
            # Apply a final zero-command policy cycle before stopping.
            run_policy_cycle(
              backend=backend,
              config=config,
              policy=policy,
              observation_builder=observation_builder,
              action_processor=action_processor,
              command=VelocityCommand(),
            )
            viewer_handle.sync()
            print(f"\n{error}")
            break

          if control.exit_requested:
            run_policy_cycle(
              backend=backend,
              config=config,
              policy=policy,
              observation_builder=observation_builder,
              action_processor=action_processor,
              command=VelocityCommand(),
            )
            viewer_handle.sync()
            break

          if control.reset_requested:
            backend.reset()
            observation_builder.reset()
            next_cycle_time = time.perf_counter()
            print("Simulation reset.")
            continue

          run_policy_cycle(
            backend=backend,
            config=config,
            policy=policy,
            observation_builder=observation_builder,
            action_processor=action_processor,
            command=control.command,
          )
          viewer_handle.sync()

          now = time.perf_counter()
          if now >= next_status_time:
            state = backend.read_state()
            base_z = backend.base_position_world[2]
            enabled_text = "ENABLED" if control.enabled else "SAFE"
            print(
              f"\r{enabled_text:7s}  "
              f"cmd=({control.command.vx:+.2f}, "
              f"{control.command.vy:+.2f}, "
              f"{control.command.wz:+.2f})  "
              f"vel=({state.base_lin_vel[0]:+.2f}, "
              f"{state.base_lin_vel[1]:+.2f}, "
              f"{state.base_ang_vel[2]:+.2f})  "
              f"z={base_z:.3f} m",
              end="",
              flush=True,
            )
            next_status_time = now + 0.25

          next_cycle_time += config.policy_period_s
          sleep_time = next_cycle_time - time.perf_counter()
          if sleep_time > 0.0:
            time.sleep(sleep_time)
          else:
            next_cycle_time = time.perf_counter()
  finally:
    backend.close()

  print("\nSpot PS5 teleoperation finished.")


if __name__ == "__main__":
  main()
