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

from gs_agent.wrappers.teleop_wrapper import TeleopWrapper
from gs_env.sim.envs.manipulation.so101_cube_env import SO101CubeEnv

import time

import genesis as gs
import torch


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
        # Create teleop wrapper first (without environment)
        print("Creating teleop wrapper...")
        env = SO101CubeEnv()
        teleop_wrapper = TeleopWrapper(
            env=env,
            device=torch.device("cpu"),
            movement_speed=0.01,  # Position movement speed
            rotation_speed=0.05   # Rotation speed
        )
        teleop_wrapper.start()


        print("Environment initialized successfully.")

        print("\n" + "="*50)
        print("SO101 TELEOP SYSTEM READY")
        print("="*50)
        
        # Run the main control loop in the main thread (Genesis viewer requires this)
        try:
            step_count = 0
            start_time = time.time()
            N_PRINT = 100
            while teleop_wrapper.running:
                # Step the teleop wrapper (this processes input and steps environment)
                teleop_wrapper.step(torch.tensor([]))
                step_count += 1

                # Print status every 1000 steps
                if step_count % N_PRINT == 0:
                    FPS = N_PRINT / (time.time() - start_time)
                    gs.logger.info(f"Running... Step {step_count}, FPS: {FPS}")
                    start_time = time.time()

                # Check for quit command
                if (teleop_wrapper.last_command and 
                    hasattr(teleop_wrapper.last_command, 'quit_teleop') and 
                    teleop_wrapper.last_command.quit_teleop):
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
