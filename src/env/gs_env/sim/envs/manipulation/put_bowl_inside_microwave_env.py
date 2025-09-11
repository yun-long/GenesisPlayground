import random
from typing import Any

import genesis as gs
import torch

from gs_env.common.bases.base_env import BaseEnv
from gs_env.sim.envs.config.schema import EnvArgs
from gs_env.sim.robots.config.schema import EEPoseAbsAction
from gs_env.sim.robots.manipulators import FrankaRobot

_DEFAULT_DEVICE = torch.device("cpu")


class PutBowlInsideMicrowaveEnv(BaseEnv):
    """Put bowl inside microwave environment."""

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

        # Table
        table_pos = args.env_config.get("table_pos", (0.0, 0.0, 0.05))
        table_size = args.env_config.get("table_size", (0.6, 0.6, 0.1))
        self.entities["table"] = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=table_pos,
                size=table_size,
            ),
        )

        # Bowl (using the winter_bowl.glb from assets)
        bowl_scale = args.env_config.get("bowl_scale", 1 / 5000)
        self.entities["bowl"] = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="assets/winter_bowl.glb",
                pos=(0.05, -0.2, 0.15),
                euler=(90, 0, 90),
                scale=bowl_scale,
                collision=True,
            ),
        )

        # Microwave (using the 7310 URDF from assets)
        microwave_pos = args.env_config.get("microwave_pos", (0.2, 0.2, 0.18))
        microwave_euler = args.env_config.get("microwave_euler", (0, 0, 30))
        microwave_scale = args.env_config.get("microwave_scale", 0.3)
        self.entities["microwave"] = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file="assets/7310/mobility.urdf",
                pos=microwave_pos,
                euler=microwave_euler,
                scale=microwave_scale,
                collision=True,
                merge_fixed_links=True,
                convexify=False,
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
        # Set bowl mass
        self.entities["bowl"].set_mass(0.01)

        # Set microwave door damping
        # Set damping for microwave (8 DOFs: 3 pos + 4 quat + 1 joint)
        self.entities["microwave"].set_dofs_damping([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0])

    def _randomize_objects(self) -> None:
        """Randomize object positions."""
        # Randomize bowl position on table
        bowl_x = random.uniform(-0.1, 0.1)
        bowl_y = random.uniform(-0.2, 0.0)
        bowl_pos = torch.tensor([bowl_x, bowl_y, 0.15], dtype=torch.float32)
        self.entities["bowl"].set_pos(bowl_pos)

        # Microwave position is fixed
        microwave_pos = torch.tensor([0.2, 0.2, 0.18], dtype=torch.float32)
        self.entities["microwave"].set_pos(microwave_pos)

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

    def get_observations(self) -> dict[str, Any]:
        """Get current observation as dictionary (BaseEnv requirement)."""
        return {
            "ee_pose": self.entities["robot"].ee_pose,
            "joint_positions": self.entities["robot"].joint_positions,
            "bowl_pos": self.entities["bowl"].get_pos(),
            "bowl_quat": self.entities["bowl"].get_quat(),
            "microwave_pos": self.entities["microwave"].get_pos(),
            "microwave_quat": self.entities["microwave"].get_quat(),
        }

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
        """Check if episode is complete - bowl is inside microwave."""
        # Get AABB (Axis-Aligned Bounding Box) for both objects
        bowl_aabb = self.entities["bowl"].get_AABB()  # [2, 3] - min and max corners
        microwave_aabb = self.entities["microwave"].get_AABB()  # [2, 3] - min and max corners

        # Check if bowl is inside microwave using AABB intersection
        # Bowl is inside if its AABB is completely contained within microwave AABB
        bowl_min, bowl_max = bowl_aabb[0], bowl_aabb[1]
        microwave_min, microwave_max = microwave_aabb[0], microwave_aabb[1]

        # Check if bowl is inside microwave (with some tolerance)
        tolerance = 0.05  # 5cm tolerance
        is_inside = torch.all(
            (bowl_min >= microwave_min - tolerance) & (bowl_max <= microwave_max + tolerance)
        )

        return torch.tensor([is_inside])

    def reset_idx(self, envs_idx: Any) -> None:
        """Reset environment."""
        # Clear any existing debug objects
        self.scene.clear_debug_objects()

        # Reset robot to natural pose
        initial_q = torch.tensor(
            [0.0, -0.3, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32
        )  # 7 joints to match registry format
        self.entities["robot"].reset_to_pose(initial_q)

        # Reset microwave door to closed position
        current_qpos = self.entities["microwave"].get_qpos()
        # Keep position and orientation, but set door joint to 0 (closed)
        reset_qpos = current_qpos.clone()
        reset_qpos[0, 7] = 0.0  # Set door joint to closed position
        self.entities["microwave"].set_qpos(reset_qpos)

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
