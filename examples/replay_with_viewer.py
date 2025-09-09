#!/usr/bin/env python3
"""
Simple trajectory replay script using teleop wrapper.
"""

import argparse

import genesis as gs
import torch
from gs_agent.wrappers.teleop_wrapper import KeyboardWrapper
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.manipulation.pick_cube_env import PickCubeEnv


def main() -> None:
    """Replay trajectory using teleop wrapper."""
    parser = argparse.ArgumentParser(description="Replay recorded robot trajectories")
    parser.add_argument(
        "--trajectory-file",
        type=str,
        help="Specific trajectory file to replay (e.g., franka_pick_place_1757368298.pkl)",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="franka",
        help="Robot type for filename prefix (default: franka)",
    )
    args = parser.parse_args()

    print("Initializing Franka Replay System...")

    # Initialize Genesis
    gs.init(
        seed=0,
        precision="32",
        logging_level="info",
        backend=gs.cpu,  # type: ignore
    )

    # Create teleop wrapper and environment
    teleop_wrapper = KeyboardWrapper(
        env=None,  # Initialize with None
        device=torch.device("cpu"),
        movement_speed=0.01,
        rotation_speed=0.05,
        trajectory_filename_prefix=f"{args.robot_type}_pick_place_",
    )
    env = PickCubeEnv(args=EnvArgsRegistry["pick_cube_default"], device=torch.device("cpu"))
    teleop_wrapper.set_environment(env)  # Set env using new method

    # Replay trajectory
    if args.trajectory_file:
        print(f"🎬 Replaying specific trajectory: {args.trajectory_file}")
        teleop_wrapper.replay_trajectory(args.trajectory_file)
    else:
        print("🎬 Replaying latest trajectory...")
        teleop_wrapper.replay_trajectory()

    print("✅ Replay completed!")
    print("👋 Closing viewer...")


if __name__ == "__main__":
    main()
