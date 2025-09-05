#!/usr/bin/env python3
"""
Simple trajectory replay script using teleop wrapper.
"""

import sys
import os
import genesis as gs

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent.gs_agent.wrappers.teleop_wrapper import TeleopWrapper
from env.gs_env.sim.envs.so101_cube_env import SO101CubeEnv


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

    # Replay latest trajectory
    teleop_wrapper.replay_latest_trajectory()

    print("✅ Replay completed!")


if __name__ == "__main__":
    main()