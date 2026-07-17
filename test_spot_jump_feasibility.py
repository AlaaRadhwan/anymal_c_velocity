#!/usr/bin/env python3
"""Visual open-loop jump feasibility test for the Spot MuJoCo model.

The robot repeatedly performs:

    stand -> crouch -> rapid extension -> landing pose -> stand

This does not use the trained walking policy or reinforcement learning.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


LEGS = ("fl", "fr", "hl", "hr")
JOINTS = ("hx", "hy", "kn")

# Existing standing pose from spot.xml.
HOME = {
    "hx": 0.0,
    "hy": 1.04,
    "kn": -1.80,
}

# Compressed pose before takeoff.
CROUCH = {
    "hx": 0.0,
    "hy": 1.52,
    "kn": -2.58,
}

# Rapid leg-extension target.
EXTEND = {
    "hx": 0.0,
    "hy": 0.35,
    "kn": -0.58,
}

# Compressed pose used to absorb the landing.
LAND = {
    "hx": 0.0,
    "hy": 1.24,
    "kn": -2.10,
}

PHYSICS_DT = 0.002
CONTROL_DT = 0.020


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display repeated Spot jump attempts in MuJoCo."
    )

    parser.add_argument(
        "--xml",
        type=Path,
        default=Path(
            "src/anymal_c_velocity/spot/xmls/scene.xml"
        ),
        help="Path to the Spot scene XML.",
    )

    parser.add_argument(
        "--torque",
        type=float,
        default=120.0,
        help=(
            "Torque limit for hip-pitch and knee actuators in N·m. "
            "This is a simulation test parameter."
        ),
    )

    parser.add_argument(
        "--depth",
        type=float,
        default=1.0,
        help="Crouch depth between 0 and 1.",
    )

    parser.add_argument(
        "--extension-time",
        type=float,
        default=0.10,
        help="Time used to extend the legs, in seconds.",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Perform only one attempt instead of repeating.",
    )

    return parser.parse_args()


def get_object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if object_id < 0:
        raise RuntimeError(
            f"MuJoCo object was not found: {name}"
        )

    return object_id


def smoothstep(value: float) -> float:
    """Smooth interpolation from zero to one."""

    value = float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )

    return value * value * (3.0 - 2.0 * value)


def interpolate_pose(
    pose_a: dict[str, float],
    pose_b: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    alpha = smoothstep(alpha)

    return {
        joint: (
            (1.0 - alpha) * pose_a[joint]
            + alpha * pose_b[joint]
        )
        for joint in JOINTS
    }


def find_actuators(
    model: mujoco.MjModel,
) -> dict[str, int]:
    actuator_ids: dict[str, int] = {}

    for leg in LEGS:
        for joint in JOINTS:
            name = f"{leg}_{joint}"

            actuator_ids[name] = get_object_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )

    return actuator_ids


def command_pose(
    data: mujoco.MjData,
    actuator_ids: dict[str, int],
    pose: dict[str, float],
) -> None:
    """Apply the same sagittal pose to all four legs."""

    for leg in LEGS:
        for joint in JOINTS:
            actuator_name = f"{leg}_{joint}"

            data.ctrl[
                actuator_ids[actuator_name]
            ] = pose[joint]


def set_torque_limits(
    model: mujoco.MjModel,
    actuator_ids: dict[str, int],
    sagittal_torque: float,
) -> None:
    """Replace the XML's ±1000 N·m limit with explicit test limits."""

    for actuator_name, actuator_id in actuator_ids.items():
        # The hip-abduction joints should mainly keep the legs lateral.
        if actuator_name.endswith("_hx"):
            torque_limit = 35.0
        else:
            torque_limit = sagittal_torque

        model.actuator_forcelimited[
            actuator_id
        ] = 1

        model.actuator_forcerange[
            actuator_id,
            0,
        ] = -torque_limit

        model.actuator_forcerange[
            actuator_id,
            1,
        ] = torque_limit


def reset_to_home(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """Reset to the home keyframe defined in spot.xml."""

    home_key_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_key_id,
    )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


def run_jump_attempt(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    actuator_ids: dict[str, int],
    depth: float,
    extension_time: float,
) -> bool:
    """Run one complete jump sequence."""

    target_crouch = {
        joint: (
            HOME[joint]
            + depth
            * (
                CROUCH[joint]
                - HOME[joint]
            )
        )
        for joint in JOINTS
    }

    # Phase timing relative to the beginning of the attempt.
    settle_end = 1.00
    crouch_end = 1.65
    crouch_hold_end = 1.85

    extension_end = (
        crouch_hold_end
        + extension_time
    )

    extension_hold_end = (
        extension_end
        + 0.18
    )

    landing_end = (
        extension_hold_end
        + 0.45
    )

    recovery_end = (
        landing_end
        + 1.50
    )

    start_sim_time = float(
        data.time
    )

    next_control_time = start_sim_time
    previous_wall_time = time.perf_counter()

    while viewer.is_running():
        local_time = (
            float(data.time)
            - start_sim_time
        )

        if local_time >= recovery_end:
            return True

        # Update target joint positions at 50 Hz.
        if float(data.time) + 1e-12 >= next_control_time:
            if local_time < settle_end:
                pose = HOME

            elif local_time < crouch_end:
                pose = interpolate_pose(
                    HOME,
                    target_crouch,
                    (
                        local_time
                        - settle_end
                    )
                    / (
                        crouch_end
                        - settle_end
                    ),
                )

            elif local_time < crouch_hold_end:
                pose = target_crouch

            elif local_time < extension_end:
                pose = interpolate_pose(
                    target_crouch,
                    EXTEND,
                    (
                        local_time
                        - crouch_hold_end
                    )
                    / extension_time,
                )

            elif local_time < extension_hold_end:
                pose = EXTEND

            elif local_time < landing_end:
                pose = interpolate_pose(
                    EXTEND,
                    LAND,
                    (
                        local_time
                        - extension_hold_end
                    )
                    / (
                        landing_end
                        - extension_hold_end
                    ),
                )

            else:
                pose = interpolate_pose(
                    LAND,
                    HOME,
                    (
                        local_time
                        - landing_end
                    )
                    / (
                        recovery_end
                        - landing_end
                    ),
                )

            command_pose(
                data,
                actuator_ids,
                pose,
            )

            next_control_time += CONTROL_DT

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()

        # Keep the simulation close to real time.
        elapsed_wall_time = (
            time.perf_counter()
            - previous_wall_time
        )

        remaining_time = (
            model.opt.timestep
            - elapsed_wall_time
        )

        if remaining_time > 0.0:
            time.sleep(
                remaining_time
            )

        previous_wall_time = time.perf_counter()

    return False


def main() -> None:
    args = parse_args()

    if not args.xml.exists():
        raise FileNotFoundError(
            f"Could not find:\n{args.xml}\n\n"
            "Run the script from the anymal_c_velocity project root."
        )

    if not 0.0 <= args.depth <= 1.0:
        raise ValueError(
            "--depth must be between 0 and 1."
        )

    if args.extension_time <= 0.0:
        raise ValueError(
            "--extension-time must be positive."
        )

    model = mujoco.MjModel.from_xml_path(
        str(
            args.xml.resolve()
        )
    )

    model.opt.timestep = PHYSICS_DT

    data = mujoco.MjData(
        model
    )

    actuator_ids = find_actuators(
        model
    )

    set_torque_limits(
        model,
        actuator_ids,
        args.torque,
    )

    reset_to_home(
        model,
        data,
    )

    body_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "body",
    )

    print()
    print("Visual jump test")
    print(f"Torque limit:   {args.torque:.1f} N·m")
    print(f"Crouch depth:   {args.depth:.2f}")
    print(f"Extension time: {args.extension_time:.3f} s")
    print("Close the MuJoCo window to stop.")
    print()

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        # Camera follows Spot automatically.
        viewer.cam.type = (
            mujoco.mjtCamera.mjCAMERA_TRACKING
        )

        viewer.cam.trackbodyid = body_id
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0

        while viewer.is_running():
            reset_to_home(
                model,
                data,
            )

            viewer.sync()

            completed = run_jump_attempt(
                model=model,
                data=data,
                viewer=viewer,
                actuator_ids=actuator_ids,
                depth=args.depth,
                extension_time=args.extension_time,
            )

            if args.once or not completed:
                break

            # Pause briefly before resetting for the next attempt.
            pause_start = time.perf_counter()

            while (
                viewer.is_running()
                and time.perf_counter() - pause_start < 0.6
            ):
                viewer.sync()
                time.sleep(0.01)


if __name__ == "__main__":
    main()