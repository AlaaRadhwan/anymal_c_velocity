"""Construct observations in the exact order used during training."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import PolicyConfig
from .robot_types import PolicyAction, RobotState, VelocityCommand


class ObservationBuilder:
  """Build the 48-value actor observation and track previous actions."""

  def __init__(self, config: PolicyConfig) -> None:
    self._config = config
    self._default_joint_positions = np.asarray(
      config.default_joint_positions,
      dtype=np.float32,
    )
    self._previous_action = np.zeros(
      config.action_size,
      dtype=np.float32,
    )

  @property
  def previous_action(self) -> NDArray[np.float32]:
    """Return a copy of the action stored for the next observation."""

    return self._previous_action.copy()

  def reset(self) -> None:
    """Reset the action history at episode or simulation reset."""

    self._previous_action.fill(0.0)

  def update_action(self, action: PolicyAction) -> None:
    """Store the latest normalized action for the next policy step."""

    if action.values.size != self._config.action_size:
      raise ValueError(
        f"Policy action must contain {self._config.action_size} values; "
        f"got {action.values.size}."
      )

    self._previous_action[:] = action.values

  def build(
    self,
    state: RobotState,
    command: VelocityCommand,
  ) -> NDArray[np.float32]:
    """Return one observation with shape ``(1, observation_size)``."""

    if state.joint_pos.size != self._config.action_size:
      raise ValueError(
        f"joint_pos must contain {self._config.action_size} values; "
        f"got {state.joint_pos.size}."
      )

    if state.joint_vel.size != self._config.action_size:
      raise ValueError(
        f"joint_vel must contain {self._config.action_size} values; "
        f"got {state.joint_vel.size}."
      )

    values = {
      "base_lin_vel": state.base_lin_vel,
      "base_ang_vel": state.base_ang_vel,
      "projected_gravity": state.projected_gravity,
      "joint_pos": state.joint_pos - self._default_joint_positions,
      "joint_vel": state.joint_vel,
      "actions": self._previous_action,
      "command": command.as_array(),
    }

    observation = np.empty(
      self._config.observation_size,
      dtype=np.float32,
    )

    for term in self._config.observation_order:
      value = values[term.name]
      if value.size != term.size:
        raise RuntimeError(
          f"Observation term {term.name!r} must contain {term.size} "
          f"values; got {value.size}."
        )
      observation[term.start:term.end] = value.reshape(-1)

    if not np.all(np.isfinite(observation)):
      raise RuntimeError("Constructed observation contains invalid values.")

    return observation[np.newaxis, :]
