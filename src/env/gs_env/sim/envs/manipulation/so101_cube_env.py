import random
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

import genesis as gs
from gs_env.common.bases.base_env import BaseEnv
from gs_env.sim.robots.so101_robot import SO101Robot


class SO101CubeEnv: # please change it to class SO101CubeEnv(BaseEnv):
    """SO101 robot environment with cube manipulation task."""

    def __init__(self) -> None:
        FPS = 60
        # Create Genesis scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                substeps=1,
                dt=1/FPS,
            ),
            rigid_options=gs.options.RigidOptions(
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 0.0, 0.7),
                camera_lookat=(0.2, 0.0, 0.1),
                camera_fov=50,
                max_FPS=200,
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
            morph=gs.morphs.Box(
                pos=(0.5, 0.0, 0.07),
                size=(0.04, 0.04, 0.04),
            ),
        )

        # Target sphere removed - using Genesis debug sphere instead

        # Build scene
        self.scene.build()

        # Initialize robot
        self.entities["robot"].initialize()

        # Command handling
        self.last_command = None

        # Store entities for easy access
        self.robot = self.entities["robot"]
        
        # Initialize target location (on ground plane)
        self.target_location = np.array([0.4, 0.0, 0.0])
        
        # Track current target point for visualization
        self.current_target_pos = None

    def initialize(self) -> None:
        """Initialize the environment."""
        # Clear any existing debug objects
        self.scene.clear_debug_objects()
        
        # Set initial robot pose
        initial_q = np.array([0.0, -0.3, 0.5, 0.0, 0.0, 0.0])
        self.entities["robot"].reset_to_pose(initial_q)

        # Randomize cube position (this will set new target location and draw debug sphere)
        self._randomize_cube()


    def apply_action(self, action: torch.Tensor) -> None:
        """Apply action to the environment (BaseEnv requirement)."""
        # This is called by the BaseEnv interface, but we use apply_command for teleop
        pass

    def apply_command(self, command: Any) -> None:
        """Apply command to the environment."""
        self.last_command = command

        # Apply command to robot
        self.entities["robot"].apply_teleop_command(command)

        # Handle special commands
        if command.reset_scene:
            self.reset_idx(torch.IntTensor([0]))
        elif command.quit_teleop:
            print("Quit command received from teleop")
        
        # Step the scene after applying command (like goal_reaching_env)
        self.scene.step()


    def get_observation(self) -> dict[str, Any] | None:
        """Get current observation from the environment."""
        robot_obs = self.entities["robot"].get_observation()

        if robot_obs is None:
            return None

        # Get cube position
        cube_pos = np.array(self.entities["cube"].get_pos())
        cube_quat = np.array(self.entities["cube"].get_quat())

        # Create observation
        observation = {
            'joint_positions': robot_obs['joint_positions'],
            'end_effector_pos': robot_obs['end_effector_pos'],
            'end_effector_quat': robot_obs['end_effector_quat'],
            'cube_pos': cube_pos,
            'cube_quat': cube_quat,
            'rgb_images': {},  # No cameras in this simple setup
            'depth_images': {}  # No depth sensors in this simple setup
        }

        return observation

    def is_episode_complete(self) -> torch.Tensor:
        """Check if episode is complete."""
        return torch.tensor([False])  # Episodes don't end automatically

    def reset_idx(self, envs_idx: Any) -> None:
        """Reset environment."""
        # Clear any existing debug objects
        self.scene.clear_debug_objects()
        
        # Reset robot to natural pose
        initial_q = np.array([0.0, -0.3, 0.5, 0.0, 0.0, 0.0])
        self.entities["robot"].reset_to_pose(initial_q)

        # Randomize cube position (this will set new target location and draw debug sphere)
        self._randomize_cube()



    def _randomize_cube(self) -> None:
        """Randomize cube position for new episodes."""
        # Ensure cube and target are far enough apart to avoid auto-success
        max_attempts = 10
        for attempt in range(max_attempts):
            cube_pos = (
                random.uniform(0.2, 0.4),
                random.uniform(-0.2, 0.2),
                0.05
            )
            cube_quat = R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat()
            
            # Set debug sphere to target location (where cube should be placed)
            target_pos = np.array([
                random.uniform(0.3, 0.5),  # Different from cube spawn location
                random.uniform(-0.3, 0.3),
                0.0  # Always on ground plane
            ])
            
            # Check distance between cube and target (only x,y coordinates)
            cube_xy = np.array(cube_pos[:2])
            target_xy = target_pos[:2]
            distance = np.linalg.norm(cube_xy - target_xy)
            
            # Ensure minimum distance of 15cm to avoid auto-success
            if distance > 0.15:
                self.entities["cube"].set_pos(cube_pos)
                self.entities["cube"].set_quat(cube_quat)
                self.target_location = target_pos
                self._draw_target_visualization(target_pos)
                return
        
        # Fallback: if we can't find a good position after max_attempts, use fixed positions
        print("⚠️  Warning: Could not find suitable cube/target positions, using fallback")
        cube_pos = (0.25, 0.0, 0.05)
        target_pos = np.array([0.45, 0.0, 0.0])
        self.entities["cube"].set_pos(cube_pos)
        self.entities["cube"].set_quat([1, 0, 0, 0])
        self.target_location = target_pos
        self._draw_target_visualization(target_pos)

    def set_target_location(self, position: NDArray[np.float64]) -> None:
        """Set the target location for cube placement."""
        # Ensure z coordinate is always 0 (on ground plane)
        target_pos = position.copy()
        target_pos[2] = 0.0
        self.target_location = target_pos
        self._draw_target_visualization(target_pos)
    
    def _draw_target_visualization(self, position: NDArray[np.float64]) -> None:
        """Draw the target sphere visualization using Genesis debug sphere."""
        # Draw debug sphere for the current target point
        self.scene.draw_debug_sphere(
            pos=position,
            radius=0.015,  # Slightly larger for better visibility
            color=(1, 0, 0, 0.8)  # Red, semi-transparent
        )
        
        # Track the current target position
        self.current_target_pos = position.copy()

    def _check_success_condition(self) -> None:
        """Check if cube is placed on target location and reset if successful."""
        # Get current cube position
        cube_pos = np.array(self.entities["cube"].get_pos())
        
        # Calculate distance between cube and target (only x,y coordinates)
        cube_xy = cube_pos[:2]
        target_xy = self.target_location[:2]
        distance = np.linalg.norm(cube_xy - target_xy)
        
        # Success threshold: cube within 5cm of target
        success_threshold = 0.05
        
        if distance < success_threshold:
            print(f"🎯 SUCCESS! Cube placed on target (distance: {distance:.3f}m)")
            print("🔄 Resetting scene...")
            # Reset the scene
            self.reset_idx(None)


    def step(self) -> None:
        """Step the simulation."""
        # Check for success condition (cube placed on target)
        self._check_success_condition()
        
        # Step Genesis simulation
        self.scene.step()


    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return 1  # Single environment for teleop

    @property
    def device(self) -> torch.device:
        """Device for tensors."""
        return gs.device
