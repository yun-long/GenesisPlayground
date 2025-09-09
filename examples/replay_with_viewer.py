#!/usr/bin/env python3
"""
Simple trajectory replay script using teleop wrapper.
"""

import genesis as gs
import torch
from gs_agent.wrappers.teleop_wrapper import KeyboardWrapper
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.manipulation.pick_cube_env import PickCubeEnv


def main() -> None:
    """Replay the latest trajectory using teleop wrapper."""
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
    )
    env = PickCubeEnv(args=EnvArgsRegistry["pick_cube_default"], device=torch.device("cpu"))
    teleop_wrapper.set_environment(env)  # Set env using new method

    # Replay the latest trajectory
    teleop_wrapper.replay_latest_trajectory()

    print("✅ Replay completed!")
    print("👋 Closing viewer...")


if __name__ == "__main__":
    main()
