"""Convert normalized policy actions into joint-position targets."""

from __future__ import annotations

import numpy as np

from .config import PolicyConfig
from .robot_types import JointTargets, PolicyAction


class ActionProcessor:
  """Apply the action scales and nominal standing joint positions."""

  def __init__(self, config: PolicyConfig) -> None:
    self._config = config
    self._defaults = np.asarray(
      config.default_joint_positions,
      dtype=np.float32,
    )
    self._scales = np.asarray(
      config.action_scales,
      dtype=np.float32,
    )

  def process(self, action: PolicyAction) -> JointTargets:
    """Return absolute targets in ``config.joint_order``."""

    if action.values.size != self._config.action_size:
      raise ValueError(
        f"Policy action must contain {self._config.action_size} values; "
        f"got {action.values.size}."
      )

    positions = self._defaults + self._scales * action.values
    if not np.all(np.isfinite(positions)):
      raise RuntimeError("Joint targets contain NaN or infinite values.")

    return JointTargets(positions)
