#!/usr/bin/env python3
"""
Teleop script for Hang Lifebuoy environment.
"""

import genesis as gs
import torch
from gs_agent.wrappers.teleop_wrapper import KeyboardWrapper
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.manipulation.hang_lifebuoy_env import HangLifebuoyEnv


def main() -> None:
    """Run teleop for hang lifebuoy task."""
    print("Initializing Hang Lifebuoy Teleop System...")

    # Initialize Genesis
    gs.init(
        seed=0,
        precision="32",
        logging_level="info",
        backend=gs.cpu,  # type: ignore
    )

    # Create teleop wrapper first (without environment)
    print("Creating teleop wrapper...")
    teleop_wrapper = KeyboardWrapper(
        env=None,
        device=torch.device("cpu"),
        movement_speed=0.01,  # Position movement speed
        rotation_speed=0.05,  # Rotation speed
        trajectory_filename_prefix="hang_lifebuoy_",
    )

    # Start teleop wrapper (keyboard listener) FIRST, before creating Genesis scene
    teleop_wrapper.start()  # type: ignore

    # Create task environment AFTER teleop wrapper is running
    env = HangLifebuoyEnv(
        args=EnvArgsRegistry["hang_lifebuoy_default"],
        device=torch.device("cpu"),
    )
    teleop_wrapper.set_environment(env)

    print("\n" + "=" * 50)
    print("Hang Lifebuoy TELEOP SYSTEM READY")
    print("=" * 50)
    print("📝 TRAJECTORY RECORDING INSTRUCTIONS:")
    print("   1. Press 'r' to start recording (anytime)")
    print("   2. Move robot with arrow keys, n/m, space")
    print("   3. Press 'r' again to stop recording and save")
    print("   💡 Recording works from any state now!")
    print("=" * 50)

    # Run the main control loop in the main thread (Genesis viewer requires this)
    try:
        step_count = 0
        while teleop_wrapper.running:
            # Step the teleop wrapper (this processes input and steps environment)
            teleop_wrapper.step(torch.tensor([]))
            step_count += 1

            # Check for quit command
            if (
                hasattr(teleop_wrapper, "last_command")
                and teleop_wrapper.last_command
                and hasattr(teleop_wrapper.last_command, "quit_teleop")
                and teleop_wrapper.last_command.quit_teleop
            ):
                print("Quit command received, exiting...")
                break

            # Safety check - exit after 1 hour of running
            if step_count > 180000:  # 1 hour at 50Hz
                print("Maximum runtime reached, exiting...")
                break

    except KeyboardInterrupt:
        print("\n👋 Teleop interrupted by user")

    finally:
        # Cleanup
        teleop_wrapper.stop()
        print("✅ Teleop session ended")


if __name__ == "__main__":
    main()
