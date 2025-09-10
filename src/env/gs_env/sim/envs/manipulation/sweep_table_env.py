import random
from typing import Any

import genesis as gs
import torch

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
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__(device=device)
        self._device = device
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
            show_viewer=True,  # Enable viewer for visualization
            show_FPS=False,
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

        # Table
        self.entities["table"] = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.05),
                size=(0.6, 0.6, 0.1),
            ),
        )

        # Broom (using a simple box as placeholder since we don't have broom.glb)
        self.entities["broom"] = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.05, -0.2, 0.15),
                size=(0.01, 0.2, 0.01),
            ),
        )

        # Trashbox A
        self.entities["trashbox_a"] = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.15, 0.0, 0.15),
                size=(0.03, 0.03, 0.03),
            ),
        )

        # Trashbox B
        self.entities["trashbox_b"] = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.15, -0.1, 0.15),
                size=(0.03, 0.03, 0.03),
            ),
        )

        # Target zone (red area on table)
        self.entities["target_zone"] = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.1, 0.3, 0.045),
                size=(0.3, 0.3, 0.003),
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

    def apply_action(self, action: torch.Tensor | Any) -> None:
        """Apply action to the environment (BaseEnv requirement)."""
        # For teleop, action might be a command object instead of tensor
        if isinstance(action, torch.Tensor):
            # Empty tensor from teleop wrapper - no action to apply
            pass
        else:
            # This is a command object from teleop
            self.last_command = action

            pos_quat = torch.concat([action.position, action.orientation], -1)
            self.entities["ee_frame"].set_qpos(pos_quat)
            # Apply action to robot

            robot_action = EEPoseAbsAction(
                ee_link_pos=action.position,
                ee_link_quat=action.orientation,
                gripper_width=0.0 if action.gripper_close else 0.04,
            )
            self.entities["robot"].apply_action(robot_action)

            # Handle special commands
            if hasattr(action, "reset_scene") and action.reset_scene:
                self.reset_idx(torch.IntTensor([0]))
            elif hasattr(action, "quit_teleop") and action.quit_teleop:
                print("Quit command received from teleop")

        # Step the scene (like goal_reaching_env)
        self.scene.step()

    def get_observations(self) -> torch.Tensor:
        """Get current observation as tensor (BaseEnv requirement)."""
        ee_pose = self.entities["robot"].ee_pose
        joint_pos = self.entities["robot"].joint_positions

        # Get object positions and orientations
        broom_pos = self.entities["broom"].get_pos()
        broom_quat = self.entities["broom"].get_quat()

        trashbox_a_pos = self.entities["trashbox_a"].get_pos()
        trashbox_a_quat = self.entities["trashbox_a"].get_quat()

        trashbox_b_pos = self.entities["trashbox_b"].get_pos()
        trashbox_b_quat = self.entities["trashbox_b"].get_quat()

        target_zone_pos = self.entities["target_zone"].get_pos()
        target_zone_quat = self.entities["target_zone"].get_quat()

        # Concatenate all observations into a single tensor
        obs_tensor = torch.cat(
            [
                ee_pose,  # 7 values: [x, y, z, qw, qx, qy, qz]
                joint_pos,  # 7 values: joint positions
                broom_pos,  # 3 values: broom position
                broom_quat,  # 4 values: broom quaternion [w, x, y, z]
                trashbox_a_pos,  # 3 values: trashbox_a position
                trashbox_a_quat,  # 4 values: trashbox_a quaternion [w, x, y, z]
                trashbox_b_pos,  # 3 values: trashbox_b position
                trashbox_b_quat,  # 4 values: trashbox_b quaternion [w, x, y, z]
                target_zone_pos,  # 3 values: target_zone position
                target_zone_quat,  # 4 values: target_zone quaternion [w, x, y, z]
            ],
            dim=-1,
        )

        return obs_tensor

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
