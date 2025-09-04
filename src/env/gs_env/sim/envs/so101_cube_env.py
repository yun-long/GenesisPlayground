import random
import time
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import genesis as gs
from gs_env.common.devices.teleop_wrapper import TeleopCommand, TeleopObservation
from gs_env.sim.robots.so101_robot import SO101Robot


class SO101CubeEnv:
    """SO101 robot environment with cube manipulation task."""

    def __init__(self) -> None:
        # Create Genesis scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(substeps=4),
            rigid_options=gs.options.RigidOptions(
                enable_joint_limit=True,
                enable_collision=True,
                gravity=(0, 0, -9.8),
                box_box_detection=True,
                constraint_timeconst=0.02,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 0.0, 0.7),
                camera_lookat=(0.2, 0.0, 0.1),
                camera_fov=50,
                max_FPS=60,
            ),
            show_viewer=True,  # Enable viewer for visualization
            show_FPS=False,
        )

        # Add entities
        self.entities = {}

        # Ground plane
        self.entities["plane"] = self.scene.add_entity(gs.morphs.Plane())

        # SO101 robot
        self.entities["robot"] = SO101Robot(self.scene)

        # Interactive cube
        self.entities["cube"] = self.scene.add_entity(
            material=gs.materials.Rigid(rho=300),
            morph=gs.morphs.Box(
                pos=(0.5, 0.0, 0.07),
                size=(0.04, 0.04, 0.04),
            ),
            surface=gs.surfaces.Default(color=(0.5, 1, 0.5)),
        )

        # Target visualization
        self.entities["target"] = self.scene.add_entity(
            gs.morphs.Mesh(
                file="meshes/axis.obj",
                scale=0.15,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
        )

        # Build scene
        self.scene.build()

        # Initialize robot
        self.entities["robot"].initialize()

        # Teleop integration
        self.teleop_wrapper = None
        self.last_command = None
        self.command_received = False

        # Store entities for easy access
        self.robot = self.entities["robot"]

    def initialize(self) -> None:
        """Initialize the environment."""

        # Set initial robot pose
        initial_q = np.array([0.0, -0.3, 0.5, 0.0, 0.0, 0.0])
        self.entities["robot"].reset_to_pose(initial_q)

        # Update target visualization
        self._update_target_visualization()

        # Randomize cube position
        self._randomize_cube()


    def apply_teleop_command(self, command: TeleopCommand) -> None:
        """Apply teleop command from teleop wrapper."""
        self.last_command = command
        self.command_received = True

        # Apply command to robot
        self.entities["robot"].apply_teleop_command(command)

        # Handle special commands
        if command.reset_scene:
            self.reset_idx(None)
        elif command.quit_teleop:
            print("Quit command received from teleop")


    def get_teleop_observation(self) -> TeleopObservation | None:
        """Get observation for teleop wrapper."""
        robot_obs = self.entities["robot"].get_observation()

        if robot_obs is None:
            return None

        # Get cube position (not used in current implementation)
        # try:
        #     cube_pos = np.array(self.entities["cube"].get_pos())
        #     cube_quat = np.array(self.entities["cube"].get_quat())
        # except Exception:
        #     # Cube position not available, use defaults
        #     pass

        # Create teleop observation
        observation = TeleopObservation(
            joint_positions=robot_obs['joint_positions'],
            end_effector_pos=robot_obs['end_effector_pos'],
            end_effector_quat=robot_obs['end_effector_quat'],
            rgb_images={},  # No cameras in this simple setup
            depth_images={}  # No depth sensors in this simple setup
        )

        return observation

    def is_episode_complete(self) -> torch.Tensor:
        """Check if episode is complete."""
        return torch.tensor([False])  # Episodes don't end automatically

    def reset_idx(self, envs_idx: Any) -> None:
        """Reset environment."""
        # Reset robot to natural pose
        initial_q = np.array([0.0, -0.3, 0.5, 0.0, 0.0, 0.0])
        self.entities["robot"].reset_to_pose(initial_q)

        # Update target visualization
        self._update_target_visualization()

        # Randomize cube position
        self._randomize_cube()

        # Update teleop wrapper if connected
        if self.teleop_wrapper:
            self._update_teleop_pose()

    def _update_target_visualization(self) -> None:
        """Update target visualization to match robot end-effector."""
        try:
            pos, quat = self.entities["robot"].get_ee_pose()
            if pos is not None:
                self.entities["target"].set_qpos(np.concatenate([pos, quat]))
        except Exception as e:
            print(f"Failed to update target visualization: {e}")

    def _randomize_cube(self) -> None:
        """Randomize cube position for new episodes."""
        try:
            cube_pos = (
                random.uniform(0.2, 0.4),
                random.uniform(-0.2, 0.2),
                0.05
            )
            cube_quat = R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat()
            self.entities["cube"].set_pos(cube_pos)
            self.entities["cube"].set_quat(cube_quat)
        except Exception as e:
            print(f"Failed to randomize cube: {e}")

    def _update_teleop_pose(self) -> None:
        """Update teleop wrapper with current robot pose."""
        try:
            pos, quat = self.entities["robot"].get_ee_pose()
            if pos is not None:
                rot = R.from_quat(quat)
                orientation = rot.as_euler('xyz')
                if self.teleop_wrapper:
                    self.teleop_wrapper.update_current_pose(pos, orientation)
        except Exception as e:
            print(f"Failed to update teleop pose: {e}")

    def set_teleop_wrapper(self, teleop_wrapper: Any) -> None:
        """Set teleop wrapper for bidirectional communication."""
        self.teleop_wrapper = teleop_wrapper

        # Set up callbacks
        if self.teleop_wrapper:
            self.teleop_wrapper.set_command_callback(self.apply_teleop_command)
            # Update teleop wrapper with robot's current pose
            if hasattr(self.entities["robot"], 'update_teleop_pose'):
                self.entities["robot"].update_teleop_pose(self.teleop_wrapper)

    def step(self) -> None:
        """Step the simulation."""
        # Process teleop input in main thread
        if self.teleop_wrapper:
            should_continue = self.teleop_wrapper.process_input()
            if not should_continue:
                return  # Signal to quit

        # Update target visualization to follow robot
        self._update_target_visualization()

        # Step Genesis simulation
        self.scene.step()

        # Reset command received flag
        if self.command_received:
            self.command_received = False

    def run_teleop_session(self, teleop_wrapper: Any) -> None:
        """Run teleop session with the provided wrapper."""
        # Set up teleop integration
        self.set_teleop_wrapper(teleop_wrapper)

        # Note: teleop_wrapper is already started in main()

        print("SO101 Cube Environment ready for teleop!")
        print("Use keyboard controls to move the robot.")
        print("Press ESC to quit.")

        try:
            # Main simulation loop
            step_count = 0
            while teleop_wrapper.running:
                self.step()
                step_count += 1

                # Print status every 1000 steps
                if step_count % 1000 == 0:
                    print(f"Running... Step {step_count}")

                # Check for quit command
                if (self.last_command and
                    hasattr(self.last_command, 'quit_teleop') and
                    self.last_command.quit_teleop):
                    print("Quit command received, exiting...")
                    break

                # Safety check - exit after 1 hour of running
                if step_count > 180000:  # 1 hour at 50Hz
                    print("Maximum runtime reached, exiting...")
                    break

                # Small delay to control simulation frequency
                time.sleep(0.02)  # 50 Hz

        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"Error in teleop session: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            teleop_wrapper.stop()
            print("Teleop session ended.")

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return 1  # Single environment for teleop

    @property
    def device(self) -> torch.device:
        """Device for tensors."""
        return gs.device
