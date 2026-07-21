"""Verify the exported policy contract without starting a simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .action_processor import ActionProcessor
from .config import PolicyConfig
from .observation_builder import ObservationBuilder
from .policy_runtime import TorchScriptPolicy
from .robot_types import RobotState, VelocityCommand


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


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Load the exported Spot policy and run one nominal inference."
  )
  parser.add_argument(
    "--policy",
    type=Path,
    default=_DEFAULT_POLICY,
    help=f"TorchScript policy path. Default: {_DEFAULT_POLICY}",
  )
  parser.add_argument(
    "--metadata",
    type=Path,
    default=_DEFAULT_METADATA,
    help=f"Policy metadata path. Default: {_DEFAULT_METADATA}",
  )
  parser.add_argument(
    "--device",
    default="cpu",
    help="Torch inference device. Default: cpu",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  config = PolicyConfig.from_json(args.metadata)
  policy = TorchScriptPolicy(args.policy, config, device=args.device)
  observation_builder = ObservationBuilder(config)
  action_processor = ActionProcessor(config)

  nominal_joint_positions = np.asarray(
    config.default_joint_positions,
    dtype=np.float32,
  )

  state = RobotState(
    base_lin_vel=np.zeros(3, dtype=np.float32),
    base_ang_vel=np.zeros(3, dtype=np.float32),
    projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
    joint_pos=nominal_joint_positions,
    joint_vel=np.zeros(config.action_size, dtype=np.float32),
  )

  observation = observation_builder.build(
    state,
    VelocityCommand(),
  )
  action = policy.act(observation)
  targets = action_processor.process(action)

  observation_builder.update_action(action)
  next_observation = observation_builder.build(
    state,
    VelocityCommand(),
  )
  action_term = config.observation_term("actions")
  stored_action = next_observation[0, action_term.start:action_term.end]

  if not np.array_equal(stored_action, action.values):
    raise RuntimeError(
      "Previous-action history was not inserted into the next observation."
    )

  print("Spot policy smoke test passed.")
  print(f"  task:              {config.task_id}")
  print(f"  policy frequency:  {config.policy_frequency_hz:.1f} Hz")
  print(f"  observation shape: {observation.shape}")
  print(f"  action shape:      {action.values.shape}")
  print(
    "  action range:      "
    f"[{action.values.min():.6f}, {action.values.max():.6f}]"
  )
  print(
    "  target range:      "
    f"[{targets.positions.min():.6f}, {targets.positions.max():.6f}] rad"
  )


if __name__ == "__main__":
  main()
