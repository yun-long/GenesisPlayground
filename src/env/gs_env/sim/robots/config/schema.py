from __future__ import annotations

from typing import TypeAlias

import torch
from gs_schemas.base_types import GenesisEnum, genesis_pydantic_config
from pydantic import BaseModel


class RigidMaterialArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    rho: float
    friction: float | None
    needs_coup: bool
    coup_friction: float
    coup_softness: float
    coup_restitution: float
    sdf_cell_size: float
    sdf_min_res: int
    sdf_max_res: int
    gravity_compensation: float


class URDFMorphArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    # Morph
    pos: tuple[float, float, float]
    euler: tuple[int, int, int]
    quat: tuple[float, float, float, float] | None
    visualization: bool
    collision: bool
    requires_jac_and_IK: bool
    is_free: bool

    # FileMorph
    file: str
    scale: float | tuple[float, float, float]
    convexify: bool
    recompute_inertia: bool

    # URDF
    fixed: bool
    prioritize_urdf_material: bool
    merge_fixed_links: bool
    links_to_keep: list[str]

    decimate: bool


class MJCFMorphArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    file: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]


class CtrlType(GenesisEnum):
    # Arm Control Types
    JOINT_POSITION = "JOINT_POSITION"
    EE_POSE_ABS = "EE_POSE_ABS"  # Absolute pose
    EE_POSE_REL = "EE_POSE_REL"  # Delta pose from current EE pose
    # Legged Control Types
    DR_JOINT_POSITION = "DR_JOINT_POSITION"  # Noised position control for legged robots


class IKSolver(GenesisEnum):
    GS = "GS"  # Genesis Solver
    PIN = "PIN"  # Pinocchio Solver


class ManipulatorRobotArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    material_args: RigidMaterialArgs
    morph_args: URDFMorphArgs | MJCFMorphArgs
    visualize_contact: bool
    vis_mode: str
    ctrl_type: CtrlType
    ik_solver: IKSolver
    ee_link_name: str
    show_target: bool
    gripper_link_names: list[str]
    default_arm_dof: dict[str, float]
    default_gripper_dof: dict[str, float] | None = None


class DomainRandomizationArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    kp_range: tuple[float, float]
    kd_range: tuple[float, float]
    motor_strength_range: tuple[float, float]
    motor_offset_range: tuple[float, float]
    friction_range: tuple[float, float]
    mass_range: tuple[float, float]
    com_displacement_range: tuple[float, float]


class LeggedRobotArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    material_args: RigidMaterialArgs
    morph_args: URDFMorphArgs
    dr_args: DomainRandomizationArgs
    visualize_contact: bool
    vis_mode: str
    ctrl_type: CtrlType
    body_link_name: str
    show_target: bool
    dof_names: list[str]
    default_dof: dict[str, float]
    soft_dof_pos_range: float
    dof_kp: dict[str, float]
    dof_kd: dict[str, float]
    action_scale: float
    ctrl_freq: int
    decimation: int


class QuadrupedRobotArgs(LeggedRobotArgs): ...


class HumanoidRobotArgs(LeggedRobotArgs):
    left_foot_link_name: str
    right_foot_link_name: str


class JointPosAction(BaseModel):
    model_config = genesis_pydantic_config(frozen=True, arbitrary_types_allowed=True)

    gripper_width: float
    joint_pos: torch.Tensor  # (n_dof,)


class EEPoseAbsAction(BaseModel):
    model_config = genesis_pydantic_config(frozen=True, arbitrary_types_allowed=True)

    gripper_width: float
    ee_link_pos: torch.Tensor  # (3,)
    ee_link_quat: torch.Tensor  # (4,)


class EEPoseRelAction(BaseModel):
    model_config = genesis_pydantic_config(frozen=True, arbitrary_types_allowed=True)

    gripper_width: float
    ee_link_pos_delta: torch.Tensor  # (3,)
    ee_link_ang_delta: torch.Tensor  # (3,)


class DRJointPosAction(BaseModel):
    model_config = genesis_pydantic_config(frozen=True, arbitrary_types_allowed=True)

    joint_pos: torch.Tensor  # (n_dof,)


BaseAction: TypeAlias = JointPosAction | EEPoseAbsAction | EEPoseRelAction | DRJointPosAction
