import random
from typing import Any

import genesis as gs
import torch
from gs_agent.wrappers.teleop_wrapper import KeyboardCommand

from gs_env.common.bases.base_env import BaseEnv
from gs_env.sim.envs.config.schema import EnvArgs
from gs_env.sim.robots.config.schema import EEPoseAbsAction
from gs_env.sim.robots.manipulators import FrankaRobot

_DEFAULT_DEVICE = torch.device("cpu")


class HangLifebuoyEnv(BaseEnv):
    """Hang lifebuoy on hanger environment."""

    def __init__(
        self,
        args: EnvArgs,
        num_envs: int = 1,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__(device=device)
        self._num_envs = num_envs
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

        # Add camera for image capture
        self._setup_camera()

        # Table
        table_pos = args.env_config.get("table_pos", (0.0, 0.0, 0.05))
        table_size = args.env_config.get("table_size", (0.6, 0.6, 0.1))
        self.entities["table"] = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=table_pos,
                size=table_size,
            ),
        )

        # Lifebuoy (using the lifebuoy.glb from assets)
        lifebuoy_scale = args.env_config.get("lifebuoy_scale", 0.03)
        self.entities["lifebuoy"] = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="assets/lifebuoy.glb",
                pos=(0.05, 0.2, 0.15),
                euler=(0, 0, 90),
                scale=lifebuoy_scale,
                collision=True,
            ),
        )

        # Hanger (using the hanger.glb from assets)
        hanger_pos_raw = args.env_config.get("hanger_pos", (0.05, -0.2, 0.15))
        hanger_euler_raw = args.env_config.get("hanger_euler", (90, 0, 90))
        hanger_scale_raw = args.env_config.get("hanger_scale", (10, 5, 10))
        hanger_pos = (
            tuple(hanger_pos_raw)
            if isinstance(hanger_pos_raw, list | tuple)
            else (0.05, -0.2, 0.15)
        )
        hanger_euler = (
            tuple(hanger_euler_raw) if isinstance(hanger_euler_raw, list | tuple) else (90, 0, 90)
        )
        hanger_scale = (
            tuple(hanger_scale_raw) if isinstance(hanger_scale_raw, list | tuple) else (10, 5, 10)
        )
        self.entities["hanger"] = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="assets/hanger.glb",
                pos=hanger_pos,
                euler=hanger_euler,
                scale=hanger_scale,
                collision=True,
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

        # Initialize with randomized positions
        self._randomize_objects()

        # Track current target point for visualization
        self.current_target_pos = None

    def _setup_camera(
        self,
        pos: tuple[float, float, float] = (1.5, 0.0, 0.7),
        lookat: tuple[float, float, float] = (0.2, 0.0, 0.1),
        fov: int = 50,
        resolution: tuple[int, int] = (640, 480),
    ) -> None:
        """Setup camera for image capture using Genesis camera renderer."""
        # Add camera to scene (Genesis-specific)
        self.camera = self.scene.add_camera(
            res=resolution,
            pos=pos,  # Camera position
            lookat=lookat,  # Camera lookat point
            fov=fov,  # Field of view
            GUI=False,  # Don't show in GUI
        )

    def get_rgb_image(self, normalize: bool = True) -> torch.Tensor | None:
        """Capture RGB image from camera."""
        try:
            # Render camera image (Genesis-specific)
            rgb, _, _, _ = self.camera.render(
                rgb=True, depth=False, segmentation=False, normal=False
            )

            # Use base class processing logic
            return self._process_rgb_tensor(rgb, normalize)
        except Exception as e:
            print(f"Warning: Could not capture camera image: {e}")
            return None

    def initialize(self) -> None:
        """Initialize the environment."""
        # Set lifebuoy mass
        self.entities["lifebuoy"].set_mass(0.01)

    def _randomize_objects(self) -> None:
        """Randomize object positions."""
        # Randomize lifebuoy position on table
        lifebuoy_x = random.uniform(-0.1, 0.1)
        lifebuoy_y = random.uniform(0.1, 0.3)
        lifebuoy_pos = torch.tensor([lifebuoy_x, lifebuoy_y, 0.15], dtype=torch.float32)
        self.entities["lifebuoy"].set_pos(lifebuoy_pos)

        # Hanger position is fixed
        hanger_pos = torch.tensor([0.05, -0.2, 0.15], dtype=torch.float32)
        self.entities["hanger"].set_pos(hanger_pos)

    def apply_action(self, action: torch.Tensor | KeyboardCommand) -> None:
        """Apply action to the environment (BaseEnv requirement)."""
        # Skip empty tensors from teleop wrapper
        if isinstance(action, torch.Tensor) and action.numel() == 0:
            return

        # Apply command object from teleop
        self.last_command = action

        # Type narrowing: at this point, action is a KeyboardCommand
        if isinstance(action, KeyboardCommand):
            pos_quat = torch.concat([action.position, action.orientation], -1)
            self.entities["ee_frame"].set_qpos(pos_quat)
            # Apply action to robot

            robot_action = EEPoseAbsAction(
                ee_link_pos=action.position,
                ee_link_quat=action.orientation,
                gripper_width=0.0 if action.gripper_close else 0.04,
            )
            self.entities["robot"].apply_action(robot_action)

        # Handle special commands (if needed in the future)
        # Note: KeyboardCommand doesn't currently support reset_scene or quit_teleop

        # Step the scene (like goal_reaching_env)
        self.scene.step()

    def get_observations(self) -> dict[str, Any]:
        """Get current observation as dictionary (BaseEnv requirement)."""
        observations = {
            "ee_pose": self.entities["robot"].ee_pose,
            "joint_positions": self.entities["robot"].joint_positions,
            "lifebuoy_pos": self.entities["lifebuoy"].get_pos(),
            "lifebuoy_quat": self.entities["lifebuoy"].get_quat(),
            "hanger_pos": self.entities["hanger"].get_pos(),
            "hanger_quat": self.entities["hanger"].get_quat(),
        }

        # Add RGB images using base class helper
        return self._add_rgb_to_observations(observations)

    def get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get end-effector pose for teleop wrapper."""
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
        """Get reward."""
        return torch.tensor([0.0]), {}

    def is_episode_complete(self) -> torch.Tensor:
        """Check if episode is complete - lifebuoy is hanging on hanger."""
        # Get AABB (Axis-Aligned Bounding Box) for both objects
        lifebuoy_aabb = self.entities["lifebuoy"].get_AABB()  # [2, 3] - min and max corners
        hanger_aabb = self.entities["hanger"].get_AABB()  # [2, 3] - min and max corners

        # Check if lifebuoy is in contact with hanger using AABB intersection
        # Lifebuoy is hanging if its AABB intersects with hanger AABB
        lifebuoy_min, lifebuoy_max = lifebuoy_aabb[0], lifebuoy_aabb[1]
        hanger_min, hanger_max = hanger_aabb[0], hanger_aabb[1]

        # Check if lifebuoy is in contact with hanger (with some tolerance)
        tolerance = 0.05  # 5cm tolerance
        is_contacting = torch.all(
            (lifebuoy_min <= hanger_max + tolerance) & (lifebuoy_max >= hanger_min - tolerance)
        )

        return torch.tensor([is_contacting])

    def reset_idx(self, envs_idx: Any) -> None:
        """Reset environment."""
        # Clear any existing debug objects
        self.scene.clear_debug_objects()

        # Reset robot to natural pose
        initial_q = torch.tensor(
            [0.0, -0.3, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32
        )  # 7 joints to match registry format
        self.entities["robot"].reset_to_pose(initial_q)

        # Randomize object positions
        self._randomize_objects()

        # Reset target visualization
        self.current_target_pos = None

    def step(self) -> None:
        """Step the environment."""
        self.scene.step()

    @property
    def num_envs(self) -> int:
        """Get number of environments."""
        return self._num_envs
