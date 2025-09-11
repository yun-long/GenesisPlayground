import random
from typing import Any

import genesis as gs
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from gs_env.common.bases.base_env import BaseEnv
from gs_env.sim.envs.config.schema import EnvArgs
from gs_env.sim.robots.config.schema import EEPoseAbsAction
from gs_env.sim.robots.manipulators import FrankaRobot

_DEFAULT_DEVICE = torch.device("cpu")


class PickCubeEnv(BaseEnv):
    """Pick cube environment."""

    def __init__(
        self,
        args: EnvArgs,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__(device=device)
        self._num_envs = 1  # Single environment for teleop
        FPS = 60
        # Create Genesis scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                substeps=4,
                dt=1 / FPS,
            ),
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
                max_FPS=200,
            ),
            show_viewer=bool(args.env_config.get("show_viewer", True)),
            show_FPS=bool(args.env_config.get("show_FPS", False)),
        )

        # Add entities
        self.entities = {}

        # Ground plane
        self.entities["plane"] = self.scene.add_entity(gs.morphs.Plane())

        # SO101 robot
        self.entities["robot"] = FrankaRobot(
            num_envs=self._num_envs,
            scene=self.scene,  # use flat scene
            args=args.robot_args,
            device=self.device,
        )

        # Interactive cube
        cube_pos = args.env_config.get("cube_pos", (0.5, 0.0, 0.07))
        cube_size = args.env_config.get("cube_size", (0.04, 0.04, 0.04))
        self.entities["cube"] = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=cube_pos,
                size=cube_size,
            ),
        )

        self.entities["ee_frame"] = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/axis.obj",
                scale=0.15,
                collision=False,
            ),
        )

        # Build scene
        self.scene.build(n_envs=1)

        # Command handling
        self.last_command = None

        # Store entities for easy access
        self.robot = self.entities["robot"]

        # Initialize with randomized cube and target positions
        self._randomize_cube()

        # Track current target point for visualization
        self.current_target_pos = None

    def initialize(self) -> None:
        """Initialize the environment."""
        # Clear any existing debug objects
        self.scene.clear_debug_objects()

        # Set initial robot pose
        initial_q = torch.tensor([0.0, -0.3, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        self.entities["robot"].reset_to_pose(initial_q)

        # Randomize cube position (this will set new target location and draw debug sphere)
        self._randomize_cube()

    # TODO: should not use Any but KeyboardCommand
    def apply_action(self, action: Any) -> None:
        """Apply action to the environment (BaseEnv requirement)."""
        # For teleop, action might be a command object instead of tensor
        if action is not None:
            # This is a command object from teleop
            self.last_command = action

            pos_quat = torch.concat([action.position, action.orientation], -1)
            self.entities["ee_frame"].set_qpos(pos_quat)
            # Apply action to robot

            action = EEPoseAbsAction(
                ee_link_pos=action.position,
                ee_link_quat=action.orientation,
                gripper_width=0.0 if action.gripper_close else 0.04,
            )
            self.entities["robot"].apply_action(action)

            # Handle special commands
            if hasattr(action, "reset_scene") and action.reset_scene:
                self.reset_idx(torch.IntTensor([0]))
            elif hasattr(action, "quit_teleop") and action.quit_teleop:
                print("Quit command received from teleop")

        # Step the scene (like goal_reaching_env)
        self.scene.step()

    def get_observations(self) -> dict[str, Any]:
        """Get current observation as dictionary (BaseEnv requirement)."""
        return {
            "ee_pose": self.entities["robot"].ee_pose,
            "joint_positions": self.entities["robot"].joint_positions,
            "cube_pos": self.entities["cube"].get_pos(),
            "cube_quat": self.entities["cube"].get_quat(),
        }

    def get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_pos = self.entities["robot"].ee_pose
        return robot_pos[..., :3], robot_pos[..., 3:]

    def get_extra_infos(self) -> dict[str, Any]:
        """Get extra information."""
        return {}

    def get_terminated(self) -> torch.Tensor:
        """Get termination status."""
        return torch.tensor([False])

    def get_truncated(self) -> torch.Tensor:
        """Get truncation status."""
        return torch.tensor([False])

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Get reward and reward components."""
        return torch.tensor([0.0]), {}

    def is_episode_complete(self) -> torch.Tensor:
        """Check if episode is complete."""
        return torch.tensor([False])  # Episodes don't end automatically

    def reset_idx(self, envs_idx: Any) -> None:
        """Reset environment."""
        # Clear any existing debug objects
        self.scene.clear_debug_objects()

        # Reset robot to natural pose
        initial_q = torch.tensor(
            [0.0, -0.3, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32
        )  # 7 joints to match registry format
        self.entities["robot"].reset_to_pose(initial_q)

        # Randomize cube position (this will set new target location and draw debug sphere)
        self._randomize_cube()

    def _randomize_cube(self) -> None:
        """Randomize cube position for new episodes."""
        # Ensure cube and target are far enough apart to avoid auto-success
        max_attempts = 10
        for _attempt in range(max_attempts):
            cube_pos = (random.uniform(0.2, 0.4), random.uniform(-0.2, 0.2), 0.05)
            cube_quat = R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat()

            # Set debug sphere to target location (where cube should be placed)
            target_pos = np.array(
                [
                    random.uniform(0.3, 0.5),  # Different from cube spawn location
                    random.uniform(-0.3, 0.3),
                    0.0,  # Always on ground plane
                ]
            )

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
            color=(1, 0, 0, 0.8),  # Red, semi-transparent
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

    # Device property inherited from BaseEnv
