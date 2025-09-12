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


class SweepTableEnv(BaseEnv):
    """Sweep table environment - sweep trashboxes to target zone."""

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

        # Broom (using the broom.glb from assets)
        broom_pos_raw = args.env_config.get("broom_pos", (0.05, -0.2, 0.15))
        broom_euler_raw = args.env_config.get("broom_euler", (90, 0, 90))
        broom_scale_raw = args.env_config.get("broom_scale", (1 / 400, 1 / 800, 1 / 400))
        broom_pos = (
            tuple(broom_pos_raw) if isinstance(broom_pos_raw, list | tuple) else (0.05, -0.2, 0.15)
        )
        broom_euler = (
            tuple(broom_euler_raw) if isinstance(broom_euler_raw, list | tuple) else (90, 0, 90)
        )
        broom_scale = (
            tuple(broom_scale_raw)
            if isinstance(broom_scale_raw, list | tuple)
            else (1 / 400, 1 / 800, 1 / 400)
        )
        self.entities["broom"] = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="assets/broom.glb",
                pos=broom_pos,
                euler=broom_euler,
                scale=broom_scale,
                collision=True,
            ),
        )

        # Trashbox A
        trashbox_size = args.env_config.get("trashbox_size", (0.03, 0.03, 0.03))
        self.entities["trashbox_a"] = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.15, 0.0, 0.15),
                size=trashbox_size,
            ),
        )

        # Trashbox B
        self.entities["trashbox_b"] = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.15, -0.1, 0.15),
                size=trashbox_size,
            ),
        )

        # Target zone (red area on table)
        target_zone_pos = args.env_config.get("target_zone_pos", (0.1, 0.3, 0.045))
        target_zone_size = args.env_config.get("target_zone_size", (0.3, 0.3, 0.003))
        self.entities["target_zone"] = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=target_zone_pos,
                size=target_zone_size,
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
        # Set object masses
        self.entities["broom"].set_mass(0.05)
        self.entities["trashbox_a"].set_mass(0.005)
        self.entities["trashbox_b"].set_mass(0.005)

    def _randomize_objects(self) -> None:
        """Randomize object positions."""
        # Randomize trashbox positions on table
        trashbox_a_x = random.uniform(0.1, 0.2)
        trashbox_a_y = random.uniform(-0.05, 0.05)
        trashbox_a_pos = torch.tensor([trashbox_a_x, trashbox_a_y, 0.15], dtype=torch.float32)
        self.entities["trashbox_a"].set_pos(trashbox_a_pos)

        trashbox_b_x = random.uniform(0.1, 0.2)
        trashbox_b_y = random.uniform(-0.15, -0.05)
        trashbox_b_pos = torch.tensor([trashbox_b_x, trashbox_b_y, 0.15], dtype=torch.float32)
        self.entities["trashbox_b"].set_pos(trashbox_b_pos)

        # Broom position is fixed
        broom_pos = torch.tensor([0.05, -0.2, 0.15], dtype=torch.float32)
        self.entities["broom"].set_pos(broom_pos)

        # Target zone position is fixed
        target_zone_pos = torch.tensor([0.1, 0.3, 0.045], dtype=torch.float32)
        self.entities["target_zone"].set_pos(target_zone_pos)

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
            "broom_pos": self.entities["broom"].get_pos(),
            "broom_quat": self.entities["broom"].get_quat(),
            "trashbox_a_pos": self.entities["trashbox_a"].get_pos(),
            "trashbox_a_quat": self.entities["trashbox_a"].get_quat(),
            "trashbox_b_pos": self.entities["trashbox_b"].get_pos(),
            "trashbox_b_quat": self.entities["trashbox_b"].get_quat(),
            "target_zone_pos": self.entities["target_zone"].get_pos(),
            "target_zone_quat": self.entities["target_zone"].get_quat(),
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
        """Check if episode is complete - both trashboxes are in target zone."""
        # Get AABB (Axis-Aligned Bounding Box) for all objects
        trashbox_a_aabb = self.entities["trashbox_a"].get_AABB()  # [2, 3] - min and max corners
        trashbox_b_aabb = self.entities["trashbox_b"].get_AABB()  # [2, 3] - min and max corners
        target_zone_aabb = self.entities["target_zone"].get_AABB()  # [2, 3] - min and max corners

        # Check if trashbox_a is in target zone
        trashbox_a_center = trashbox_a_aabb.mean(dim=0)
        target_zone_min, target_zone_max = target_zone_aabb[0], target_zone_aabb[1]

        trashbox_a_inside = torch.all(
            (trashbox_a_center >= target_zone_min) & (trashbox_a_center <= target_zone_max)
        )

        # Check if trashbox_b is in target zone
        trashbox_b_center = trashbox_b_aabb.mean(dim=0)
        trashbox_b_inside = torch.all(
            (trashbox_b_center >= target_zone_min) & (trashbox_b_center <= target_zone_max)
        )

        # Both trashboxes must be in target zone
        return torch.tensor([trashbox_a_inside and trashbox_b_inside])

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
