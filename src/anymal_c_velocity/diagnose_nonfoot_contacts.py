"""Long-duration, physics-substep contact diagnostic for Spot."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch

import anymal_c_velocity  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg


TASK_ID = "Mjlab-Velocity-Flat-Spot"

# Small thresholds used only to distinguish inactive sensor values
# from an actual contact sample.
FORCE_EPSILON_N = 1.0e-3
PENETRATION_EPSILON_M = 1.0e-7


@dataclass
class ContactStats:
    contact_samples: int = 0
    force_sum: float = 0.0
    max_force: float = 0.0
    min_distance: float = math.inf

    first_sample: int | None = None
    last_sample: int | None = None

    event_count: int = 0
    longest_event_samples: int = 0

    current_event_start: int | None = None
    current_event_max_force: float = 0.0
    current_event_min_distance: float = math.inf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seconds",
        type=float,
        default=120.0,
        help="Amount of simulated time to observe.",
    )

    parser.add_argument(
        "--spawn-z",
        type=float,
        default=0.46,
        help="Initial Spot base height in metres.",
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("spot_contacts"),
        help="Prefix for the generated CSV files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 0

    # Keep the test deterministic.
    for event_name in (
        "push_robot",
        "foot_friction",
        "base_com",
        "encoder_bias",
    ):
        cfg.events.pop(event_name, None)

    if "reset_base" in cfg.events:
        cfg.events["reset_base"].params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    # Override the initial height for controlled comparisons.
    robot_cfg = cfg.scene.entities["robot"]
    robot_cfg.init_state.pos = (0.0, 0.0, args.spawn_z)

    # Prevent resets from interrupting the measurement.
    # This is diagnostic only and must not be used for training.
    cfg.terminations.clear()

    diagnostic_sensor_cfg = ContactSensorCfg(
        name="all_ground_contacts_diag",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=(
                r".*_collision$",
                r"^(FL|FR|HL|HR)$",
            ),
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=(
            "found",
            "force",
            "dist",
        ),
        reduce="maxforce",
        num_slots=1,

        # Store all physics substeps belonging to one control step.
        history_length=cfg.decimation,
    )

    cfg.scene.sensors = (
        cfg.scene.sensors or ()
    ) + (diagnostic_sensor_cfg,)

    env = ManagerBasedRlEnv(
        cfg=cfg,
        device="cuda:0",
    )

    env.reset()

    sensor = env.scene["all_ground_contacts_diag"]

    # Resolve the exact order used by the sensor.
    # _slots is used only for this diagnostic utility.
    primary_names: list[str] = []

    for slot in sensor._slots:
        if slot.primary_name not in primary_names:
            primary_names.append(slot.primary_name)

    stats = {
        name: ContactStats()
        for name in primary_names
    }

    contact_events: list[dict[str, float | str]] = []

    zero_actions = torch.zeros(
        (
            env.num_envs,
            env.action_manager.total_action_dim,
        ),
        device=env.device,
    )

    number_of_env_steps = math.ceil(
        args.seconds / env.step_dt
    )

    total_physics_samples = 0

    print("\nRunning contact diagnostic")
    print("=" * 70)
    print(f"Task:              {TASK_ID}")
    print(f"Spawn height:      {args.spawn_z:.3f} m")
    print(f"Requested time:    {args.seconds:.2f} s")
    print(f"Physics timestep:  {env.physics_dt:.6f} s")
    print(f"Control timestep:  {env.step_dt:.6f} s")
    print(f"Decimation:        {cfg.decimation}")
    print(f"Tracked geoms:     {len(primary_names)}")
    print("=" * 70)

    for _ in range(number_of_env_steps):
        env.step(zero_actions)

        force_history = sensor.data.force_history
        distance_history = sensor.data.dist_history

        if force_history is None:
            raise RuntimeError(
                "Contact force history was not created."
            )

        if distance_history is None:
            raise RuntimeError(
                "Contact distance history was not created."
            )

        # Shapes:
        # force_history:    [env, geom, substep, xyz]
        # distance_history: [env, geom, substep]
        #
        # History index 0 is newest, so reverse it to process
        # the physics samples chronologically.
        forces = torch.flip(
            force_history[0],
            dims=(1,),
        ).detach().cpu()

        distances = torch.flip(
            distance_history[0],
            dims=(1,),
        ).detach().cpu()

        force_magnitudes = torch.linalg.vector_norm(
            forces,
            dim=-1,
        )

        number_of_substeps = force_magnitudes.shape[1]

        for substep_index in range(number_of_substeps):
            sample_index = total_physics_samples

            for geom_index, geom_name in enumerate(primary_names):
                force = float(
                    force_magnitudes[
                        geom_index,
                        substep_index,
                    ]
                )

                distance = float(
                    distances[
                        geom_index,
                        substep_index,
                    ]
                )

                contact_active = (
                    force > FORCE_EPSILON_N
                    or distance < -PENETRATION_EPSILON_M
                )

                geom_stats = stats[geom_name]

                if contact_active:
                    geom_stats.contact_samples += 1
                    geom_stats.force_sum += force
                    geom_stats.max_force = max(
                        geom_stats.max_force,
                        force,
                    )
                    geom_stats.min_distance = min(
                        geom_stats.min_distance,
                        distance,
                    )

                    if geom_stats.first_sample is None:
                        geom_stats.first_sample = sample_index

                    geom_stats.last_sample = sample_index

                    if geom_stats.current_event_start is None:
                        geom_stats.current_event_start = sample_index
                        geom_stats.current_event_max_force = force
                        geom_stats.current_event_min_distance = distance
                    else:
                        geom_stats.current_event_max_force = max(
                            geom_stats.current_event_max_force,
                            force,
                        )
                        geom_stats.current_event_min_distance = min(
                            geom_stats.current_event_min_distance,
                            distance,
                        )

                elif geom_stats.current_event_start is not None:
                    start_sample = geom_stats.current_event_start
                    end_sample = sample_index

                    duration_samples = end_sample - start_sample

                    geom_stats.event_count += 1
                    geom_stats.longest_event_samples = max(
                        geom_stats.longest_event_samples,
                        duration_samples,
                    )

                    contact_events.append(
                        {
                            "geom": geom_name,
                            "start_s": (
                                start_sample * env.physics_dt
                            ),
                            "end_s": (
                                end_sample * env.physics_dt
                            ),
                            "duration_s": (
                                duration_samples * env.physics_dt
                            ),
                            "max_force_N": (
                                geom_stats.current_event_max_force
                            ),
                            "min_distance_m": (
                                geom_stats.current_event_min_distance
                            ),
                        }
                    )

                    geom_stats.current_event_start = None
                    geom_stats.current_event_max_force = 0.0
                    geom_stats.current_event_min_distance = math.inf

            total_physics_samples += 1

    # Close any contact event that remains active when testing ends.
    for geom_name, geom_stats in stats.items():
        if geom_stats.current_event_start is None:
            continue

        start_sample = geom_stats.current_event_start
        end_sample = total_physics_samples
        duration_samples = end_sample - start_sample

        geom_stats.event_count += 1
        geom_stats.longest_event_samples = max(
            geom_stats.longest_event_samples,
            duration_samples,
        )

        contact_events.append(
            {
                "geom": geom_name,
                "start_s": start_sample * env.physics_dt,
                "end_s": end_sample * env.physics_dt,
                "duration_s": duration_samples * env.physics_dt,
                "max_force_N": geom_stats.current_event_max_force,
                "min_distance_m": (
                    geom_stats.current_event_min_distance
                ),
            }
        )

    actual_duration = (
        total_physics_samples * env.physics_dt
    )

    print("\nContact summary")
    print("=" * 145)

    header = (
        f"{'Geom':24s}"
        f"{'Contact %':>11s}"
        f"{'Total s':>11s}"
        f"{'Events':>9s}"
        f"{'Longest s':>12s}"
        f"{'First s':>11s}"
        f"{'Last s':>11s}"
        f"{'Max N':>11s}"
        f"{'Mean N':>11s}"
        f"{'Min dist mm':>14s}"
    )

    print(header)
    print("-" * 145)

    summary_rows: list[dict[str, float | int | str]] = []

    for geom_name in primary_names:
        geom_stats = stats[geom_name]

        contact_percentage = (
            100.0
            * geom_stats.contact_samples
            / total_physics_samples
        )

        total_contact_time = (
            geom_stats.contact_samples
            * env.physics_dt
        )

        longest_event_time = (
            geom_stats.longest_event_samples
            * env.physics_dt
        )

        mean_force = (
            geom_stats.force_sum
            / geom_stats.contact_samples
            if geom_stats.contact_samples > 0
            else 0.0
        )

        first_time = (
            geom_stats.first_sample * env.physics_dt
            if geom_stats.first_sample is not None
            else math.nan
        )

        last_time = (
            geom_stats.last_sample * env.physics_dt
            if geom_stats.last_sample is not None
            else math.nan
        )

        minimum_distance_mm = (
            1000.0 * geom_stats.min_distance
            if geom_stats.min_distance != math.inf
            else math.nan
        )

        print(
            f"{geom_name:24s}"
            f"{contact_percentage:11.3f}"
            f"{total_contact_time:11.3f}"
            f"{geom_stats.event_count:9d}"
            f"{longest_event_time:12.4f}"
            f"{first_time:11.3f}"
            f"{last_time:11.3f}"
            f"{geom_stats.max_force:11.3f}"
            f"{mean_force:11.3f}"
            f"{minimum_distance_mm:14.4f}"
        )

        summary_rows.append(
            {
                "geom": geom_name,
                "contact_percentage": contact_percentage,
                "total_contact_time_s": total_contact_time,
                "event_count": geom_stats.event_count,
                "longest_event_s": longest_event_time,
                "first_contact_s": first_time,
                "last_contact_s": last_time,
                "max_force_N": geom_stats.max_force,
                "mean_force_N": mean_force,
                "min_distance_mm": minimum_distance_mm,
            }
        )

    print("=" * 145)
    print(f"Measured simulated time: {actual_duration:.3f} s")
    print(f"Physics samples:         {total_physics_samples}")

    summary_path = Path(
        f"{args.output_prefix}_summary.csv"
    )

    events_path = Path(
        f"{args.output_prefix}_events.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with events_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "geom",
            "start_s",
            "end_s",
            "duration_s",
            "max_force_N",
            "min_distance_m",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(contact_events)

    print("\nSettled MjLab robot state")
    print("=" * 90)

    robot = env.scene["robot"]

    root_pose = robot.data.root_link_pose_w[0].detach().cpu()
    root_velocity = robot.data.root_link_vel_w[0].detach().cpu()

    root_position = root_pose[:3]
    root_quaternion = root_pose[3:]

    w, x, y, z = (float(value) for value in root_quaternion)

    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    pitch = math.asin(sin_pitch)

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)

    print("Root position:", root_position.numpy())
    print(
        "Root orientation [deg]:",
        {
            "roll": math.degrees(roll),
            "pitch": math.degrees(pitch),
            "yaw": math.degrees(yaw),
        },
    )
    print("Root velocity:", root_velocity.numpy())
    print(
        "Projected gravity:",
        robot.data.projected_gravity_b[0].detach().cpu().numpy(),
    )

    joint_positions = robot.data.joint_pos[0].detach().cpu()
    joint_targets = robot.data.joint_pos_target[0].detach().cpu()
    default_positions = robot.data.default_joint_pos[0].detach().cpu()
    joint_velocities = robot.data.joint_vel[0].detach().cpu()
    actuator_forces = robot.data.actuator_force[0].detach().cpu()

    print("\nJoint state:")
    print(
        f"{'Joint':10s}"
        f"{'Actual':>12s}"
        f"{'Target':>12s}"
        f"{'Default':>12s}"
        f"{'Error':>12s}"
        f"{'Velocity':>12s}"
        f"{'Act. force':>14s}"
    )
    print("-" * 84)

    for index, joint_name in enumerate(robot.joint_names):
        actual = float(joint_positions[index])
        target = float(joint_targets[index])
        default = float(default_positions[index])
        velocity = float(joint_velocities[index])
        force = float(actuator_forces[index])

        print(
            f"{joint_name:10s}"
            f"{actual:12.6f}"
            f"{target:12.6f}"
            f"{default:12.6f}"
            f"{actual - target:12.6f}"
            f"{velocity:12.6f}"
            f"{force:14.3f}"
        )

    print("\nFoot-site positions:")
    site_positions = robot.data.site_pos_w[0].detach().cpu()

    for site_name in ("FL_site", "FR_site", "HL_site", "HR_site"):
        site_index = robot.site_names.index(site_name)
        position = site_positions[site_index]

        print(
            f"{site_name:10s}",
            f"x={float(position[0]): .6f}",
            f"y={float(position[1]): .6f}",
            f"z={float(position[2]): .6f}",
        )

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved events:  {events_path}")

    env.close()


if __name__ == "__main__":
    main()