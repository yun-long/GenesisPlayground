from __future__ import annotations

from collections.abc import Callable
from typing import Any

import genesis as gs
import numpy as np
import torch
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from gymnasium import spaces

from gs_env.common.bases.base_robot import BaseGymRobot
from gs_env.common.utils.math_utils import quat_from_euler
from gs_env.sim.robots.config.schema import (
    BaseAction,
    CtrlType,
    DRJointPosAction,
    HumanoidRobotArgs,
    JointPosAction,
    LeggedRobotArgs,
    ManipulatorRobotArgs,
    QuadrupedRobotArgs,
)


class LeggedRobotBase(BaseGymRobot):
    """
    Base class for legged robots
    """

    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: LeggedRobotArgs,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        # == set members ==
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # == Genesis configurations ==
        material = gs.materials.Rigid(**args.material_args.model_dump())
        morph = gs.morphs.URDF(**args.morph_args.model_dump())
        self._robot: RigidEntity = scene.add_entity(
            material=material,
            morph=morph,
            visualize_contact=args.visualize_contact,
            vis_mode=args.vis_mode,
        )

        # == action space ==
        n_dof = len(args.default_dof.keys())
        self._action_space = spaces.Box(shape=(n_dof,), low=-np.inf, high=np.inf)
        assert self._action_space is not None, "Action space cannot be None"

        # == some buffer initialization ==
        self._init()

    def _init(self) -> None:
        self._dof_dim = len(self._args.default_dof.keys())  # total number of joints

        #
        self._dof_idx = torch.arange(self._dof_dim, device=self._device)

        #
        self._body_link = self._robot.get_link(self._args.body_link_name)
        self._body_link_idx = self._body_link.idx
        #
        self._dofs_idx_local = [
            self._robot.get_joint(name).dofs_idx_local[0] for name in self._args.dof_names
        ]
        dof_kp, dof_kd = [], []
        for dof_name in self._args.dof_names:
            for key in self._args.dof_kp.keys():
                if key in dof_name:
                    dof_kp.append(self._args.dof_kp[key])
                    dof_kd.append(self._args.dof_kd[key])
        self._dof_kp = torch.tensor(dof_kp, device=self._device)
        self._dof_kd = torch.tensor(dof_kd, device=self._device)
        self._batched_dof_kp = self._dof_kp[None, :].repeat(self._num_envs, 1)
        self._batched_dof_kd = self._dof_kd[None, :].repeat(self._num_envs, 1)
        self._motor_strength = torch.ones(
            (self._num_envs, self._dof_dim), device=self._device
        )  # motor strength scaling factor
        self._motor_offset = torch.zeros(
            (self._num_envs, self._dof_dim), device=self._device
        )  # motor offset
        self._torque = torch.zeros((self._num_envs, self._dof_dim), device=self._device)

        # default states
        self._default_pos = torch.tensor(
            self._args.morph_args.pos, dtype=torch.float32, device=self._device
        )
        self._default_euler = torch.tensor(
            self._args.morph_args.euler, dtype=torch.float32, device=self._device
        )
        self._default_quat = quat_from_euler(self._default_euler)
        default_joint_angles = [self._args.default_dof[name] for name in self._args.dof_names]
        self._default_dof_pos = torch.tensor(
            default_joint_angles, dtype=torch.float32, device=self._device
        )
        # buffers
        self._dof_pos = torch.zeros(
            (self._num_envs, self._dof_dim), dtype=torch.float32, device=self._device
        )
        self._dof_vel = torch.zeros(
            (self._num_envs, self._dof_dim), dtype=torch.float32, device=self._device
        )

        # == set up control dispatch ==
        self._dispatch: dict[CtrlType, Callable[[BaseAction], None]] = {
            CtrlType.JOINT_POSITION.value: self._apply_joint_pos,
            CtrlType.DR_JOINT_POSITION.value: self._apply_dr_joint_pos,
        }

    def post_build_init(self) -> None:
        self._init_domain_randomization()

        # limits
        self._dof_pos_limits = torch.stack(self._robot.get_dofs_limit(self._dofs_idx_local), dim=1)
        for i in range(self._dof_pos_limits.shape[0]):
            # soft limits
            m = (self._dof_pos_limits[i, 0] + self._dof_pos_limits[i, 1]) / 2
            r = self._dof_pos_limits[i, 1] - self._dof_pos_limits[i, 0]
            self._dof_pos_limits[i, 0] = m - 0.5 * r * self._args.soft_dof_pos_range
            self._dof_pos_limits[i, 1] = m + 0.5 * r * self._args.soft_dof_pos_range
        self._torque_limits = self._robot.get_dofs_force_range(self._dofs_idx_local)[1]

    def _init_domain_randomization(self) -> None:
        envs_idx: torch.IntTensor = torch.arange(0, self._num_envs, device=self._device)
        self._randomize_rigids(envs_idx)
        self._randomize_controls(envs_idx)

    def _randomize_rigids(self, envs_idx: torch.IntTensor) -> None:
        # friction
        min_friction, max_friction = self._args.dr_args.friction_range
        solver: RigidSolver = self._robot.solver
        ratios = (
            torch.rand(len(envs_idx), 1).repeat(1, solver.n_geoms) * (max_friction - min_friction)
            + min_friction
        )
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), envs_idx)
        # mass
        min_mass, max_mass = self._args.dr_args.mass_range
        added_mass = torch.rand(len(envs_idx), 1) * (max_mass - min_mass) + min_mass
        solver.set_links_mass_shift(
            added_mass,
            [
                self._body_link_idx,
            ],
            envs_idx,
        )
        # com displacement
        min_com, max_com = self._args.dr_args.com_displacement_range
        # displacement = (torch.rand(len(envs_idx), 3) - 0.5) * (max_com - min_com) + min_com
        # solver.set_links_COM_shift(displacement, [self._body_link_idx,], envs_idx)

    def _randomize_controls(self, envs_idx: torch.IntTensor) -> None:
        # kp
        min_kp, max_kp = self._args.dr_args.kp_range
        ratios = torch.rand(len(envs_idx), self._dof_dim) * (max_kp - min_kp) + min_kp
        self._batched_dof_kp[envs_idx] = ratios * self._dof_kp[None, 0]
        # self._robot.set_dofs_kp(
        #     self._batched_dof_kp[envs_idx], dofs_idx_local=self._dofs_idx_local, envs_idx=envs_idx
        # )
        # kd
        min_kd, max_kd = self._args.dr_args.kd_range
        ratios = torch.rand(len(envs_idx), self._dof_dim) * (max_kd - min_kd) + min_kd
        self._batched_dof_kd[envs_idx] = ratios * self._dof_kd[None, 0]
        # self._robot.set_dofs_kv(
        #     self._batched_dof_kd[envs_idx], dofs_idx_local=self._dofs_idx_local, envs_idx=envs_idx
        # )
        # motor strength
        min_strength, max_strength = self._args.dr_args.motor_strength_range
        self._motor_strength[envs_idx] = (
            torch.rand(len(envs_idx), self._dof_dim) * (max_strength - min_strength) + min_strength
        )
        # motor offset
        min_offset, max_offset = self._args.dr_args.motor_offset_range
        self._motor_offset[envs_idx] = (
            torch.rand(len(envs_idx), self._dof_dim) * (max_offset - min_offset) + min_offset
        )

    def reset(self, envs_idx: torch.IntTensor) -> None:
        if len(envs_idx) == 0:
            return
        self.reset_idx(envs_idx)

    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        self._robot.set_pos(
            self._default_pos[None].repeat(len(envs_idx), 1),
            envs_idx=envs_idx,
            zero_velocity=True,
        )
        self._robot.set_quat(
            self._default_quat[None].repeat(len(envs_idx), 1),
            envs_idx=envs_idx,
            zero_velocity=True,
        )
        self._robot.set_dofs_position(
            self._default_dof_pos[None].repeat(len(envs_idx), 1),
            envs_idx=envs_idx,
            dofs_idx_local=self._dofs_idx_local,
            zero_velocity=True,
        )
        self._dof_pos[envs_idx] = self._default_dof_pos[None].repeat(len(envs_idx), 1)
        self._dof_vel[envs_idx] = 0.0

    def set_state(
        self,
        pos: torch.Tensor,
        quat: torch.Tensor,
        dof_pos: torch.Tensor,
        envs_idx: torch.IntTensor,
        lin_vel: torch.Tensor | None = None,
        ang_vel: torch.Tensor | None = None,
        dof_vel: torch.Tensor | None = None,
    ) -> None:
        self._robot.set_pos(pos, envs_idx=envs_idx)
        self._robot.set_quat(quat, envs_idx=envs_idx)
        dof_pos = torch.clamp(dof_pos, self._dof_pos_limits[:, 0], self._dof_pos_limits[:, 1])
        self._robot.set_dofs_position(
            dof_pos, envs_idx=envs_idx, dofs_idx_local=self._dofs_idx_local
        )
        self._dof_pos[envs_idx] = dof_pos.clone()
        if lin_vel is not None:
            self._robot.set_dofs_velocity(lin_vel, envs_idx=envs_idx, dofs_idx_local=[0, 1, 2])
        else:
            self._robot.set_dofs_velocity(
                torch.zeros((len(envs_idx), 3), device=self._device),
                envs_idx=envs_idx,
                dofs_idx_local=[0, 1, 2],
            )
        if ang_vel is not None:
            self._robot.set_dofs_velocity(ang_vel, envs_idx=envs_idx, dofs_idx_local=[3, 4, 5])
        else:
            self._robot.set_dofs_velocity(
                torch.zeros((len(envs_idx), 3), device=self._device),
                envs_idx=envs_idx,
                dofs_idx_local=[3, 4, 5],
            )
        if dof_vel is not None:
            self._robot.set_dofs_velocity(
                dof_vel, envs_idx=envs_idx, dofs_idx_local=self._dofs_idx_local
            )
            self._dof_vel[envs_idx] = dof_vel.clone()
        else:
            self._robot.set_dofs_velocity(
                torch.zeros((len(envs_idx), self._dof_dim), device=self._device),
                envs_idx=envs_idx,
                dofs_idx_local=self._dofs_idx_local,
            )
            self._dof_vel[envs_idx] = 0.0

    def apply_action(self, action: BaseAction | torch.Tensor) -> None:
        """
        Apply the action to the robot.
        """
        if isinstance(action, torch.Tensor):
            match self.ctrl_type:
                case CtrlType.DR_JOINT_POSITION:
                    action = DRJointPosAction(joint_pos=action)
                case CtrlType.JOINT_POSITION:
                    action = JointPosAction(joint_pos=action, gripper_width=0.0)
                case _:
                    raise ValueError(f"Unsupported control type: {self.ctrl_type}")
        self._dispatch[self._args.ctrl_type](action)
        self._dof_pos[:] = self._robot.get_dofs_position(self._dofs_idx_local)
        self._dof_vel[:] = self._robot.get_dofs_velocity(self._dofs_idx_local)

    def _apply_joint_pos(self, act: JointPosAction) -> None:
        """
        Apply joint position control to the robot.
        """
        assert act.joint_pos.shape == (
            self._num_envs,
            self._dof_dim,
        ), "Joint position action must match the number of joints."
        q_target = act.joint_pos.to(self._device)
        self._robot.control_dofs_position(position=q_target)

    def _apply_dr_joint_pos(self, act: DRJointPosAction) -> None:
        """
        Apply noised joint position control to the robot.
        """
        assert act.joint_pos.shape == (
            self._num_envs,
            self._dof_dim,
        ), "Joint position action must match the number of joints."
        q_force = (
            self._batched_dof_kp
            * (act.joint_pos + self._default_dof_pos - self._dof_pos + self._motor_offset)
            - self._batched_dof_kd * self._dof_vel
        )
        q_force = q_force * self._motor_strength
        q_force = torch.clamp(q_force, -self._torque_limits, self._torque_limits)
        self._torque[:] = q_force
        self._robot.control_dofs_force(force=q_force, dofs_idx_local=self._dofs_idx_local)

    @property
    def dof_dim(self) -> int:
        return self._dof_dim

    @property
    def default_pos(self) -> torch.Tensor:
        return self._default_pos

    @property
    def default_quat(self) -> torch.Tensor:
        return self._default_quat

    @property
    def default_dof_pos(self) -> torch.Tensor:
        return self._default_dof_pos

    @property
    def base_pos(self) -> torch.Tensor:
        return self._robot.get_pos()

    @property
    def base_quat(self) -> torch.Tensor:
        return self._robot.get_quat()

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._dof_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._dof_vel

    @property
    def torque(self) -> torch.Tensor:
        return self._torque

    @property
    def dof_pos_limits(self) -> torch.Tensor:
        return self._dof_pos_limits

    def __getattr__(self, item: str) -> Any:
        if hasattr(self._robot, item):
            return getattr(self._robot, item)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    @property
    def ctrl_type(self) -> CtrlType:
        return self._args.ctrl_type


class HumanoidRobotBase(LeggedRobotBase):
    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: ManipulatorRobotArgs | QuadrupedRobotArgs | HumanoidRobotArgs,
        device: str = "cpu",
    ) -> None:
        super().__init__(num_envs, scene, args, device)


class G1Robot(HumanoidRobotBase):
    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: ManipulatorRobotArgs | QuadrupedRobotArgs | HumanoidRobotArgs,
        device: str = "cpu",
    ) -> None:
        super().__init__(num_envs, scene=scene, args=args, device=device)
