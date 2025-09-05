#!/usr/bin/env python3
"""
Simple trajectory replay script using teleop wrapper.
"""

import time
import genesis as gs

from gs_agent.wrappers.teleop_wrapper import TeleopWrapper
from gs_env.sim.envs.so101_cube_env import SO101CubeEnv


def main():
    """Replay the latest trajectory using teleop wrapper."""
    print("Initializing SO101 Replay System...")

    # Initialize Genesis
    gs.init(
        seed=0,
        precision="32",
        logging_level="info",
        backend=gs.cpu  # type: ignore
    )
    print("Genesis initialized successfully.")

    # Create teleop wrapper and environment
    teleop_wrapper = TeleopWrapper()
    env = SO101CubeEnv()

    # Set the environment in the wrapper
    teleop_wrapper.set_environment(env)

    # Replay the other trajectory (earlier one)
    teleop_wrapper.replay_latest_trajectory()

    print("✅ Replay completed!")
    print("👋 Closing viewer...")


if __name__ == "__main__":
    main()