#!/usr/bin/env python3
"""
SO101 Robot Teleop Script

This script demonstrates the modular teleop system with:
1. TeleopWrapper: Robot-agnostic keyboard input handler
2. SO101CubeEnv: Task environment with SO101 robot and cube
3. Bidirectional communication between teleop and environment

Usage:
    python src/env/gs_env/scripts/run_so101_teleop.py
"""

from gs_env.common.devices.teleop_wrapper import TeleopWrapper
from gs_env.sim.envs.so101_cube_env import SO101CubeEnv

import genesis as gs


def main() -> None:
    """Run SO101 teleop session."""
    print("Initializing SO101 Teleop System...")

    # Initialize Genesis
    gs.init(
        seed=0,
        precision="32",
        logging_level="info",
        backend=gs.cpu  # type: ignore
    )

    print("Genesis initialized successfully.")

    try:
        # Create teleop wrapper and start it FIRST (like original script)
        print("Creating teleop wrapper...")
        teleop_wrapper = TeleopWrapper(
            movement_speed=0.01,  # Position movement speed
            rotation_speed=0.05   # Rotation speed
        )

        # Start teleop wrapper BEFORE creating Genesis scene
        teleop_wrapper.start()

        # Create task environment AFTER teleop wrapper is running
        print("Creating SO101 cube environment...")
        env = SO101CubeEnv()
        env.initialize()

        print("Environment initialized successfully.")
        print("\n" + "="*50)
        print("SO101 TELEOP SYSTEM READY")
        print("="*50)

        # Run teleop session
        env.run_teleop_session(teleop_wrapper)

    except KeyboardInterrupt:
        print("\nTeleop session interrupted by user.")
    except Exception as e:
        print(f"Error during teleop session: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        # Cleanup is handled by the environment and teleop wrapper


if __name__ == "__main__":
    main()
