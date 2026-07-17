"""RL configuration for ANYmal C velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def anymal_c_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for ANYmal C velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      stochastic=True,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.01,
    ),
    experiment_name="anymal_c_velocity",
    max_iterations=10_000,
  )


def spot_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the PPO runner configuration for Spot."""

  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      stochastic=True,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.01,
    ),
    experiment_name="spot_velocity",
    max_iterations=10_000,
  )


def spot_jump_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the PPO configuration for full-cycle vertical jumping."""

  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(
        512,
        256,
        128,
      ),
      stochastic=True,

      # Keep enough exploration for rapid, unfamiliar movements.
      init_noise_std=0.35,
    ),

    critic=RslRlModelCfg(
      hidden_dims=(
        512,
        256,
        128,
      ),
    ),

    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.01,
    ),

    # Prevent the enlarged position-action scales from receiving
    # arbitrarily large raw network outputs.
    clip_actions=1.0,

    experiment_name="spot_jump",
    run_name="vertical_full_cycle",

    save_interval=100,
    max_iterations=10_000,
  )