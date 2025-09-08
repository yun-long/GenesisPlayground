from typing import Any

import numpy as np
import torch
from gymnasium import spaces

import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink  # type: ignore
from gs_env.common.bases.base_robot import BaseGymRobot
from gs_env.sim.robots.config.schema import (
    CtrlType,
    EEPoseAbsAction,
    EEPoseRelAction,
    JointPosAction,
    ManipulatorRobotArgs,
)


class ManipulatorBase(BaseGymRobot):
    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,  # type: ignore
        args: ManipulatorRobotArgs,
        device: torch.device,
    ) -> None:
        super().__init__()
        # == set members ==
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # == Genesis configurations ==
        material: gs.materials.Rigid = gs.materials.Rigid()  # type: ignore
        morph: gs.morphs.MJCF = gs.morphs.MJCF(  # type: ignore
            file=args.morph_args.file,
            pos=args.morph_args.pos,
            quat=args.morph_args.quat,
        )
        #
        robot_entity = scene.add_entity(
            material=material,
            morph=morph,
            visualize_contact=args.visualize_contact,
            vis_mode=args.vis_mode,
        )
        assert isinstance(robot_entity, RigidEntity), (
            "Robot entity must be an instance of gs.Entity"
        )
        self._robot_entity = robot_entity

        # == action space ==
        n_dof = len(args.default_arm_dof.keys())
        action_space_dict: dict[str, spaces.Space[Any]] = {"gripper_width": spaces.Box(0.0, 0.08)}
        match args.ctrl_type:
            case CtrlType.JOINT_POSITION:
                action_space_dict.update(
                    {"joint_pos": spaces.Box(shape=(n_dof,), low=-1.0, high=1.0)}
                )
            case CtrlType.EE_POSE_ABS:
                action_space_dict.update(
                    {
                        "ee_link_pos": spaces.Box(shape=(3,), low=-1.0, high=1.0),
                        "ee_link_quat": spaces.Box(shape=(4,), low=-1.0, high=1.0),
                    }
                )
            case CtrlType.EE_POSE_REL:
                action_space_dict.update(
                    {
                        "ee_link_pos_delta": spaces.Box(shape=(3,), low=-1.0, high=1.0),
                        "ee_link_ang_delta": spaces.Box(
                            shape=(3,), low=-1.0, high=1.0
                        ),  # roll, pitch, yaw
                    }
                )

        self._action_space = spaces.Dict(action_space_dict)

        # == some buffer initialization ==
        self._init()

    def _init(self) -> None:
        self._arm_dof_dim = len(self._args.default_arm_dof.keys())  # total number of joints
        self._gripper_dim = 0
        if self._args.default_gripper_dof is not None:
            self._gripper_dim = len(
                self._args.default_gripper_dof.keys()
            )  # number of gripper joints

        #
        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self._device,
        )

        #
        self._ee_link: RigidLink = self._robot_entity.get_link(self._args.ee_link_name)
        self._left_finger_link: RigidLink = self._robot_entity.get_link(
            self._args.gripper_link_names[0]
        )
        self._right_finger_link: RigidLink = self._robot_entity.get_link(
            self._args.gripper_link_names[1]
        )
        #
        self._default_joint_angles = list(self._args.default_arm_dof.values())
        if self._args.default_gripper_dof is not None:
            self._default_joint_angles += list(self._args.default_gripper_dof.values())

        # == set up control dispatch ==
        self._dispatch = {
            CtrlType.JOINT_POSITION.value: self._apply_joint_pos,
            CtrlType.EE_POSE_ABS.value: self._apply_ee_pose_abs,
            CtrlType.EE_POSE_REL.value: self._apply_ee_pose_rel,
        }

    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        if envs_idx is None or len(envs_idx) == 0:
            return
        self.go_home(envs_idx)

    def go_home(self, envs_idx: torch.IntTensor) -> None:
        default_joint_angles = torch.tensor(
            self._default_joint_angles, dtype=torch.float32, device=self._device
        ).repeat(len(envs_idx), 1)
        self._robot_entity.set_qpos(default_joint_angles, envs_idx=envs_idx)

    def reset_to_pose(
        self, joint_positions: torch.Tensor | list[float] | np.ndarray[Any, np.dtype[np.floating]]
    ) -> None:
        """Reset robot to a specific joint configuration."""
        if isinstance(joint_positions, list | np.ndarray):
            joint_positions = torch.tensor(
                joint_positions, dtype=torch.float32, device=self._device
            )

        # Ensure the tensor has the right shape (batch_size, num_joints)
        if joint_positions.dim() == 1:
            joint_positions = joint_positions.unsqueeze(0)

        # If only arm joints are provided, pad with default gripper values
        if joint_positions.shape[1] < len(self._default_joint_angles):
            arm_joints = joint_positions
            # Get gripper joint values, handling potential duplicates
            gripper_values = (
                list(self._args.default_gripper_dof.values())
                if self._args.default_gripper_dof
                else []
            )
            # Remove duplicates while preserving order
            gripper_values = list(dict.fromkeys(gripper_values))
            gripper_joints = (
                torch.tensor(gripper_values, dtype=torch.float32, device=self._device)
                .unsqueeze(0)
                .repeat(joint_positions.shape[0], 1)
            )
            joint_positions = torch.cat([arm_joints, gripper_joints], dim=1)

        # Create envs_idx for all environments
        envs_idx = torch.arange(joint_positions.shape[0], device=self._device)
        self._robot_entity.set_qpos(joint_positions, envs_idx=envs_idx)

    def apply_action(
        self, action: JointPosAction | EEPoseAbsAction | EEPoseRelAction | torch.Tensor
    ) -> None:
        """
        Apply the action to the robot.
        """
        if isinstance(action, torch.Tensor):
            match self.ctrl_type:
                case CtrlType.JOINT_POSITION:
                    action = JointPosAction(joint_pos=action, gripper_width=0.0)
                case CtrlType.EE_POSE_ABS:
                    action = EEPoseAbsAction(
                        ee_link_pos=action[..., :3],
                        ee_link_quat=action[..., 3:7],
                        gripper_width=0.0,
                    )
                case CtrlType.EE_POSE_REL:
                    action = EEPoseRelAction(
                        ee_link_pos_delta=action[..., :3],
                        ee_link_ang_delta=action[..., 3:6],
                        gripper_width=0.0,
                    )
        self._dispatch[self._args.ctrl_type](action)

    def _apply_joint_pos(self, act: JointPosAction) -> None:
        """
        Apply joint position control to the robot.
        """
        assert act.joint_pos.shape == (
            self._num_envs,
            self._arm_dof_dim,
        ), "Joint position action must match the number of joints."
        q_target = act.joint_pos.to(self._device)
        self._robot_entity.control_dofs_position(position=q_target)

    def _apply_ee_pose_abs(self, act: EEPoseAbsAction) -> None:
        """
        Apply end-effector pose control to the robot.
        """
        assert act.ee_link_pos.shape == (
            self._num_envs,
            3,
        ), "End-effector position must be a 3D vector."
        assert act.ee_link_quat.shape == (
            self._num_envs,
            4,
        ), "End-effector quaternion must be a 4D vector."

        target_pos = act.ee_link_pos.to(self._device)
        target_quat = act.ee_link_quat.to(self._device)

        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_pos,
            quat=target_quat,
            dofs_idx_local=self._arm_dof_idx,
            max_samples=10,  # number of IK samples
            max_solver_iters=20,  # maximum solver iterations
        )
        q_pos[:, self._fingers_dof] = torch.tensor(
            [act.gripper_width, act.gripper_width], device=self._device
        )
        self._robot_entity.control_dofs_position(position=q_pos)

    def _apply_ee_pose_rel(self, act: EEPoseRelAction) -> None:
        """
        Apply relative end-effector pose control to the robot.
        """
        assert act.ee_link_pos_delta.shape == (
            self._num_envs,
            3,
        ), "End-effector position delta must be a 3D vector."
        assert act.ee_link_ang_delta.shape == (
            self._num_envs,
            3,
        ), "End-effector angle delta must be a 3D vector."
        q_pos = self._dls_ik(act)
        # set gripper width
        q_pos[:, self._fingers_dof] = torch.tensor(
            [act.gripper_width, -act.gripper_width], device=self._device
        )
        self._robot_entity.control_dofs_position(position=q_pos)

    def _dls_ik(self, action: EEPoseRelAction) -> torch.Tensor:
        """
        Damped least squares inverse kinematics
        """
        delta_pose = torch.cat([action.ee_link_pos_delta, action.ee_link_ang_delta], dim=-1)
        lambda_val = 0.01
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        jacobian_T = jacobian.transpose(1, 2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
        delta_joint_pos = (
            jacobian_T
            @ torch.inverse(jacobian @ jacobian_T + lambda_matrix)
            @ delta_pose.unsqueeze(-1)
        ).squeeze(-1)
        return self._robot_entity.get_qpos() + delta_joint_pos

    @property
    def joint_positions(self) -> torch.Tensor:
        return self._robot_entity.get_qpos()

    @property
    def base_pos(self) -> torch.Tensor:
        return self._robot_entity.get_pos()

    @property
    def ee_pose(self) -> torch.Tensor:
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_finger_pose(self) -> torch.Tensor:
        pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def right_finger_pose(self) -> torch.Tensor:
        pos, quat = (
            self._right_finger_link.get_pos(),
            self._right_finger_link.get_quat(),
        )
        return torch.cat([pos, quat], dim=-1)

    def __getattr__(self, item: str) -> Any:
        # Use object.__getattribute__ to avoid recursion with hasattr
        try:
            return object.__getattribute__(self._robot_entity, item)
        except AttributeError as err:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'"
            ) from err

    @property
    def ctrl_type(self) -> CtrlType:
        return self._args.ctrl_type


class FrankaRobot(ManipulatorBase):
    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,  # type: ignore
        args: ManipulatorRobotArgs,
        device: torch.device,  # type: ignore
    ) -> None:
        super().__init__(num_envs=num_envs, scene=scene, args=args, device=device)
