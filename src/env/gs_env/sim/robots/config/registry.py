from typing import TypeAlias

from gs_env.sim.robots.config.schema import (
    CtrlType,
    IKSolver,
    ManipulatorRobotArgs,
    MJCFMorphArgs,
    RigidMaterialArgs,
    URDFMorphArgs,
)

# ------------------------------------------------------------
# Material
# ------------------------------------------------------------

MaterialArgs: TypeAlias = RigidMaterialArgs


MaterialArgsRegistry: dict[str, MaterialArgs] = {}


MaterialArgsRegistry["default"] = RigidMaterialArgs(
    rho=200.0,
    friction=None,
    needs_coup=True,
    coup_friction=0.1,
    coup_softness=0.002,
    coup_restitution=0.0,
    sdf_cell_size=0.005,
    sdf_min_res=32,
    sdf_max_res=128,
    gravity_compensation=1,
)


# ------------------------------------------------------------
# Morph
# ------------------------------------------------------------

MorphArgs: TypeAlias = URDFMorphArgs | MJCFMorphArgs


MorphArgsRegistry: dict[str, MorphArgs] = {}


MorphArgsRegistry["franka_default"] = MJCFMorphArgs(
    pos=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    file="xml/franka_emika_panda/panda.xml",
)


# ------------------------------------------------------------
# Robot
# ------------------------------------------------------------


RobotArgsRegistry: dict[str, ManipulatorRobotArgs] = {}

RobotArgsRegistry["franka_default"] = ManipulatorRobotArgs(
    material_args=MaterialArgsRegistry["default"],
    morph_args=MorphArgsRegistry["franka_default"],
    visualize_contact=False,
    vis_mode="visual",
    ctrl_type=CtrlType.EE_POSE_REL,
    ik_solver=IKSolver.GS,  # TODO: need to be aligned with real
    ee_link_name="hand",  # or "gripper_tip"
    show_target=True,
    gripper_link_names=[
        "left_finger",
        "right_finger",
    ],
    default_arm_dof={
        "joint1": 0.0,
        "joint2": -0.785,
        "joint3": 0.0,
        "joint4": -2.356,
        "joint5": 0.0,
        "joint6": 1.57,
        "joint7": 0.785,
    },
    default_gripper_dof={
        "joint7": 0.04,
        "joint8": -0.04,
    },  # open gripper
)


RobotArgsRegistry["franka_teleop"] = ManipulatorRobotArgs(
    material_args=MaterialArgsRegistry["default"],
    morph_args=MorphArgsRegistry["franka_default"],
    visualize_contact=False,
    vis_mode="visual",
    ctrl_type=CtrlType.EE_POSE_ABS,
    ik_solver=IKSolver.GS,  # TODO: need to be aligned with real
    ee_link_name="hand",  # or "gripper_tip"
    show_target=True,
    gripper_link_names=[
        "left_finger",
        "right_finger",
    ],
    default_arm_dof={
        "joint1": 0.0,
        "joint2": -0.785,
        "joint3": 0.0,
        "joint4": -2.356,
        "joint5": 0.0,
        "joint6": 1.57,
        "joint7": 0.785,
    },
    default_gripper_dof={
        "joint7": 0.04,
        "joint8": -0.04,
    },  # open gripper
)
