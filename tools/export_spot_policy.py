"""Export the trained Spot velocity policy for standalone deployment."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import onnx
import torch
from rsl_rl.runners import OnPolicyRunner

# Importing the project package registers its MjLab tasks.
import anymal_c_velocity  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


DEFAULT_TASK_ID = "Mjlab-Velocity-Rough-Spot"
DEFAULT_OUTPUT_DIR = Path("artifacts/policies/spot_velocity")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Export a Spot velocity checkpoint to TorchScript and ONNX."
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    required=True,
    help="Path to the RSL-RL checkpoint, for example model_7500.pt.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help=f"Export directory. Default: {DEFAULT_OUTPUT_DIR}",
  )
  parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Torch device used while loading the checkpoint. Default: CUDA if available, otherwise CPU.",
  )
  parser.add_argument(
    "--task",
    default=DEFAULT_TASK_ID,
    choices=(
      "Mjlab-Velocity-Rough-Spot",
      "Mjlab-Velocity-Flat-Spot",
    ),
    help=f"MjLasb task used to reconstruct the policy. Default: {DEFAULT_TASK_ID}",
  )
  return parser.parse_args()


def resolve_device(requested_device: str | None) -> str:
  if requested_device is None:
    return "cpu"

  if requested_device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(
      f"CUDA device '{requested_device}' was requested, but CUDA is unavailable."
    )

  return requested_device


def build_observation_layout(env: ManagerBasedRlEnv) -> tuple[list[dict], int]:
  manager = env.observation_manager

  if not manager.group_obs_concatenate["actor"]:
    raise RuntimeError("The actor observation group must be concatenated for export.")

  names = manager.active_terms["actor"]
  shapes = manager.group_obs_term_dim["actor"]

  layout: list[dict] = []
  start = 0

  for name, shape in zip(names, shapes, strict=True):
    size = math.prod(shape)
    end = start + size
    layout.append(
      {
        "name": name,
        "shape": list(shape),
        "start": start,
        "end": end,
      }
    )
    start = end

  return layout, start


def action_contract(env: ManagerBasedRlEnv) -> tuple[list[str], list[float], list[float]]:
  joint_action = env.action_manager.get_term("joint_pos")
  if not isinstance(joint_action, JointPositionAction):
    raise TypeError("Expected the 'joint_pos' action term to be JointPositionAction.")

  joint_ids = joint_action.target_ids
  robot = env.scene["robot"]

  joint_order = list(joint_action.target_names)
  default_joint_positions = (
    robot.data.default_joint_pos[0, joint_ids].detach().cpu().tolist()
  )

  scale = joint_action.scale
  if isinstance(scale, torch.Tensor):
    action_scales = scale[0].detach().cpu().tolist()
  else:
    action_scales = [float(scale)] * joint_action.action_dim

  return joint_order, default_joint_positions, action_scales


def validate_exports(
  runner,
  wrapped_env: RslRlVecEnvWrapper,
  jit_path: Path,
  onnx_path: Path,
  device: str,
) -> None:
  observations = wrapped_env.get_observations()

  with torch.inference_mode():
    reference_actions = runner.get_inference_policy(device=device)(observations).cpu()

    jit_policy = torch.jit.load(str(jit_path), map_location="cpu").eval()
    jit_actions = jit_policy(observations["actor"].cpu())

  torch.testing.assert_close(
    jit_actions,
    reference_actions,
    rtol=1e-5,
    atol=1e-6,
  )

  onnx.checker.check_model(onnx.load(str(onnx_path)))


def main() -> None:
  args = parse_args()
  configure_torch_backends()

  checkpoint = args.checkpoint.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  device = resolve_device(args.device)

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)

  # One environment is sufficient to resolve the policy contract and export it.
  env_cfg.scene.num_envs = 1
  env_cfg.commands["twist"].debug_vis = False

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  try:
    wrapped_env = RslRlVecEnvWrapper(
      env,
      clip_actions=agent_cfg.clip_actions,
    )

    runner_cls = load_runner_cls(args.task) or OnPolicyRunner
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )

    jit_path = output_dir / "policy_jit.pt"
    onnx_path = output_dir / "policy.onnx"
    metadata_path = output_dir / "policy_metadata.json"

    runner.export_policy_to_jit(str(output_dir), filename=jit_path.name)
    runner.export_policy_to_onnx(str(output_dir), filename=onnx_path.name)

    observation_order, observation_size = build_observation_layout(env)
    joint_order, default_joint_positions, action_scales = action_contract(env)
    action_size = env.action_manager.total_action_dim

    metadata = {
      "schema_version": 1,
      "task_id": args.task,
      "checkpoint": str(checkpoint),
      "policy_frequency_hz": 1.0 / env.step_dt,
      "physics_timestep_s": env.physics_dt,
      "control_decimation": env.cfg.decimation,
      "observation_size": observation_size,
      "action_size": action_size,
      "joint_order": joint_order,
      "default_joint_positions": default_joint_positions,
      "action_scales": action_scales,
      "observation_order": observation_order,
      "command_order": ["vx", "vy", "wz"],
      "command_units": ["m/s", "m/s", "rad/s"],
    }

    with metadata_path.open("w", encoding="utf-8") as file:
      json.dump(metadata, file, indent=2)
      file.write("\n")

    # Preserve MjLab's standard metadata inside the ONNX artifact as well.
    onnx_metadata = get_base_metadata(env, str(checkpoint))
    onnx_metadata.update(
      {
        "task_id": args.task,
        "policy_frequency_hz": metadata["policy_frequency_hz"],
        "observation_size": observation_size,
        "action_size": action_size,
      }
    )
    attach_metadata_to_onnx(str(onnx_path), onnx_metadata)

    validate_exports(runner, wrapped_env, jit_path, onnx_path, device)

    print(f"[INFO] Exported TorchScript: {jit_path}")
    print(f"[INFO] Exported ONNX:       {onnx_path}")
    print(f"[INFO] Exported metadata:   {metadata_path}")
    print(
      f"[INFO] Contract: observations={observation_size}, "
      f"actions={action_size}, frequency={metadata['policy_frequency_hz']:.1f} Hz"
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()