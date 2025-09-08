#!/usr/bin/env python3
"""
Pick Cube Teleop Script

This script demonstrates the modular teleop system with:
1. TeleopWrapper: Robot-agnostic keyboard input handler
2. PickCubeEnv: Task environment with SO101 robot and cube
3. Bidirectional communication between teleop and environment

Usage:
    python src/env/gs_env/scripts/run_pick_cube_teleop.py
"""

import genesis as gs
import torch
from gs_agent.wrappers.teleop_wrapper import KeyboardWrapper
from gs_env.common.bases.base_env import BaseEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.manipulation.pick_cube_env import PickCubeEnv


# todo teleop
# todo clean interface
def create_gs_env(env_name: str = "pick_cube_default") -> BaseEnv:
    device = torch.device("cpu")
    return PickCubeEnv(args=EnvArgsRegistry[env_name], device=device)


def main() -> None:
    """Run Pick Cube teleop session."""
    print("Initializing Pick Cube Teleop System...")

    # Initialize Genesis
    gs.init(
        seed=0,
        precision="32",
        logging_level="info",
        backend=gs.cpu,  # type: ignore
    )

    try:
        # Create teleop wrapper first (without environment)
        print("Creating teleop wrapper...")
        teleop_wrapper = KeyboardWrapper(
            env=None,
            device=torch.device("cpu"),
            movement_speed=0.01,  # Position movement speed
            rotation_speed=0.05,  # Rotation speed
        )

        # Start teleop wrapper (keyboard listener) FIRST, before creating Genesis scene
        teleop_wrapper.start()  # type: ignore

        # Create task environment AFTER teleop wrapper is running
        env = create_gs_env()
        teleop_wrapper.set_environment(env)
        print("\n" + "=" * 50)
        print("Pick Cube TELEOP SYSTEM READY")
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
                    hasattr(teleop_wrapper, 'last_command')
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
            print("\nInterrupted by user (Ctrl+C)")
        finally:
            teleop_wrapper.stop()
            print("Teleop session ended.")

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
