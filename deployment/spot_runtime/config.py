"""Load and validate the exported Spot policy contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


_EXPECTED_OBSERVATIONS: tuple[tuple[str, int], ...] = (
  ("base_lin_vel", 3),
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("joint_pos", 12),
  ("joint_vel", 12),
  ("actions", 12),
  ("command", 3),
)

_EXPECTED_COMMAND_ORDER = ("vx", "vy", "wz")


@dataclass(frozen=True)
class ObservationTerm:
  """One contiguous term in the flattened policy observation."""

  name: str
  shape: tuple[int, ...]
  start: int
  end: int

  @property
  def size(self) -> int:
    """Return the flattened size of this term."""

    return math.prod(self.shape)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "ObservationTerm":
    """Create a term from one metadata entry."""

    return cls(
      name=str(data["name"]),
      shape=tuple(int(value) for value in data["shape"]),
      start=int(data["start"]),
      end=int(data["end"]),
    )


@dataclass(frozen=True)
class PolicyConfig:
  """Validated deployment contract for one exported policy."""

  schema_version: int
  task_id: str
  checkpoint: str
  policy_frequency_hz: float
  physics_timestep_s: float
  control_decimation: int
  observation_size: int
  action_size: int
  joint_order: tuple[str, ...]
  default_joint_positions: tuple[float, ...]
  action_scales: tuple[float, ...]
  observation_order: tuple[ObservationTerm, ...]
  command_order: tuple[str, ...]
  command_units: tuple[str, ...]

  @property
  def policy_period_s(self) -> float:
    """Return the interval between two policy evaluations."""

    return 1.0 / self.policy_frequency_hz

  def observation_term(self, name: str) -> ObservationTerm:
    """Return a named observation term."""

    for term in self.observation_order:
      if term.name == name:
        return term

    raise KeyError(f"Unknown observation term: {name!r}")

  @classmethod
  def from_json(cls, path: str | Path) -> "PolicyConfig":
    """Load and validate a policy metadata JSON file."""

    metadata_path = Path(path).expanduser().resolve()
    if not metadata_path.is_file():
      raise FileNotFoundError(
        f"Policy metadata file not found: {metadata_path}"
      )

    with metadata_path.open("r", encoding="utf-8") as file:
      data = json.load(file)

    config = cls(
      schema_version=int(data["schema_version"]),
      task_id=str(data["task_id"]),
      checkpoint=str(data.get("checkpoint", "")),
      policy_frequency_hz=float(data["policy_frequency_hz"]),
      physics_timestep_s=float(data["physics_timestep_s"]),
      control_decimation=int(data["control_decimation"]),
      observation_size=int(data["observation_size"]),
      action_size=int(data["action_size"]),
      joint_order=tuple(str(name) for name in data["joint_order"]),
      default_joint_positions=tuple(
        float(value) for value in data["default_joint_positions"]
      ),
      action_scales=tuple(float(value) for value in data["action_scales"]),
      observation_order=tuple(
        ObservationTerm.from_dict(term)
        for term in data["observation_order"]
      ),
      command_order=tuple(str(name) for name in data["command_order"]),
      command_units=tuple(str(unit) for unit in data["command_units"]),
    )

    config._validate()
    return config

  def _validate(self) -> None:
    """Reject metadata that cannot safely drive this Spot policy."""

    if self.schema_version != 1:
      raise ValueError(
        f"Unsupported metadata schema_version={self.schema_version}; expected 1."
      )

    if self.policy_frequency_hz <= 0.0:
      raise ValueError("policy_frequency_hz must be positive.")

    if self.physics_timestep_s <= 0.0:
      raise ValueError("physics_timestep_s must be positive.")

    if self.control_decimation <= 0:
      raise ValueError("control_decimation must be positive.")

    expected_frequency = 1.0 / (
      self.physics_timestep_s * self.control_decimation
    )
    if not math.isclose(
      self.policy_frequency_hz,
      expected_frequency,
      rel_tol=1e-6,
      abs_tol=1e-6,
    ):
      raise ValueError(
        "Inconsistent timing contract: policy_frequency_hz does not match "
        "physics_timestep_s and control_decimation."
      )

    if self.action_size <= 0 or self.observation_size <= 0:
      raise ValueError("Policy input and output sizes must be positive.")

    if len(self.joint_order) != self.action_size:
      raise ValueError(
        "joint_order length does not match action_size."
      )

    if len(set(self.joint_order)) != len(self.joint_order):
      raise ValueError("joint_order contains duplicate joint names.")

    if len(self.default_joint_positions) != self.action_size:
      raise ValueError(
        "default_joint_positions length does not match action_size."
      )

    if len(self.action_scales) != self.action_size:
      raise ValueError("action_scales length does not match action_size.")

    if any(scale <= 0.0 for scale in self.action_scales):
      raise ValueError("Every action scale must be positive.")

    offset = 0
    observed_layout: list[tuple[str, int]] = []
    for term in self.observation_order:
      if term.start != offset:
        raise ValueError(
          f"Observation term {term.name!r} starts at {term.start}; "
          f"expected {offset}."
        )

      if term.end - term.start != term.size:
        raise ValueError(
          f"Observation term {term.name!r} has inconsistent shape and indices."
        )

      observed_layout.append((term.name, term.size))
      offset = term.end

    if offset != self.observation_size:
      raise ValueError(
        "Observation terms do not fill observation_size exactly."
      )

    if tuple(observed_layout) != _EXPECTED_OBSERVATIONS:
      raise ValueError(
        "Unexpected Spot observation layout. "
        f"Received {tuple(observed_layout)!r}."
      )

    if self.command_order != _EXPECTED_COMMAND_ORDER:
      raise ValueError(
        f"Unexpected command order {self.command_order!r}; "
        f"expected {_EXPECTED_COMMAND_ORDER!r}."
      )

    if len(self.command_units) != len(self.command_order):
      raise ValueError("command_units length does not match command_order.")
