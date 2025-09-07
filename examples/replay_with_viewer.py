#!/usr/bin/env python3
"""
Simple trajectory replay script using teleop wrapper.
"""

import genesis as gs
from gs_agent.wrappers.teleop_wrapper import TeleopWrapper
from gs_env.sim.envs.manipulation.so101_cube_env import SO101CubeEnv


def main() -> None:
    """Replay the latest trajectory using teleop wrapper."""
    print("Initializing SO101 Replay System...")

    # Initialize Genesis
    gs.init(
        seed=0,
        precision="32",
        logging_level="info",
        backend=gs.cpu,  # type: ignore
    )

    # Create teleop wrapper and environment
    teleop_wrapper = TeleopWrapper(
        env=None,  # Initialize with None
        device=gs.cpu,
        movement_speed=0.01,
        rotation_speed=0.05,
    )
    env = SO101CubeEnv()  # Create env after wrapper
    teleop_wrapper.set_environment(env)  # Set env using new method

    # Replay the other trajectory (earlier one)
    teleop_wrapper.replay_latest_trajectory()

    print("✅ Replay completed!")
    print("👋 Closing viewer...")


if __name__ == "__main__":
    main()
