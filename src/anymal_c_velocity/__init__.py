from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  anymal_c_flat_env_cfg,
  anymal_c_rough_env_cfg,
)
from .rl_cfg import anymal_c_ppo_runner_cfg, spot_ppo_runner_cfg
from .spot_env_cfgs import spot_flat_env_cfg, spot_rough_env_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Anymal-C",
  env_cfg=anymal_c_rough_env_cfg(),
  play_env_cfg=anymal_c_rough_env_cfg(play=True),
  rl_cfg=anymal_c_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Anymal-C",
  env_cfg=anymal_c_flat_env_cfg(),
  play_env_cfg=anymal_c_flat_env_cfg(play=True),
  rl_cfg=anymal_c_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Spot",
  env_cfg=spot_rough_env_cfg(),
  play_env_cfg=spot_rough_env_cfg(play=True),
  rl_cfg=spot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Spot",
  env_cfg=spot_flat_env_cfg(),
  play_env_cfg=spot_flat_env_cfg(play=True),
  rl_cfg=spot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
