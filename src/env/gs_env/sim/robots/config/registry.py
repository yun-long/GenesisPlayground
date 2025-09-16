from typing import TypeAlias

from gs_env.sim.robots.config.schema import (
    CtrlType,
    DomainRandomizationArgs,
    HumanoidRobotArgs,
    IKSolver,
    ManipulatorRobotArgs,
    MJCFMorphArgs,
    QuadrupedRobotArgs,
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


MaterialArgsRegistry["g1_default"] = RigidMaterialArgs(
    rho=200.0,
    friction=None,
    needs_coup=True,
    coup_friction=0.1,
    coup_softness=0.002,
    coup_restitution=0.0,
    sdf_cell_size=0.005,
    sdf_min_res=32,
    sdf_max_res=128,
    gravity_compensation=0,
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


MorphArgsRegistry["g1_default"] = URDFMorphArgs(
    pos=(0.0, 0.0, 0.8),
    euler=(0, 0, 0),
    quat=None,
    visualization=True,
    collision=True,
    requires_jac_and_IK=True,
    is_free=True,
    file="assets/unitree_g1/g1_custom_collision_29dof.urdf",
    scale=1.0,
    convexify=False,
    recompute_inertia=False,
    fixed=False,
    prioritize_urdf_material=False,
    merge_fixed_links=False,
    links_to_keep=[],
    decimate=True,
)


# ------------------------------------------------------------
# Domain Randomization
# ------------------------------------------------------------

DRArgs: TypeAlias = DomainRandomizationArgs


DRArgsRegistry: dict[str, DRArgs] = {}


DRArgsRegistry["default"] = DomainRandomizationArgs(
    kp_range=(0.75, 1.25),
    kd_range=(0.75, 1.25),
    motor_strength_range=(0.75, 1.25),
    motor_offset_range=(-0.05, 0.05),
    friction_range=(0.5, 1.5),
    mass_range=(-1.0, 3.0),
    com_displacement_range=(-0.05, 0.05),
)


# ------------------------------------------------------------
# Robot
# ------------------------------------------------------------


RobotArgsRegistry: dict[str, ManipulatorRobotArgs | QuadrupedRobotArgs | HumanoidRobotArgs] = {}

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


RobotArgsRegistry["g1_default"] = HumanoidRobotArgs(
    material_args=MaterialArgsRegistry["g1_default"],
    morph_args=MorphArgsRegistry["g1_default"],
    dr_args=DRArgsRegistry["default"],
    visualize_contact=False,
    vis_mode="visual",
    ctrl_type=CtrlType.DR_JOINT_POSITION,
    body_link_name="torso_link",
    left_foot_link_name="left_ankle_roll_link",
    right_foot_link_name="right_ankle_roll_link",
    show_target=True,
    dof_names=[
        # Lower body
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        # Waist
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        # Upper body
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    default_dof={
        "left_hip_pitch_joint": -0.1,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.3,
        "left_ankle_pitch_joint": -0.2,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.1,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.3,
        "right_ankle_pitch_joint": -0.2,
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    },
    soft_dof_pos_range=0.9,
    dof_kp={
        "hip_yaw": 200,
        "hip_roll": 200,
        "hip_pitch": 200,
        "knee": 200,
        "ankle_pitch": 20,
        "ankle_roll": 20,
        "waist_yaw": 400,
        "waist_roll": 400,
        "waist_pitch": 400,
        "shoulder_pitch": 90,
        "shoulder_roll": 60,
        "shoulder_yaw": 20,
        "elbow": 60,
        "wrist_roll": 4.0,
        "wrist_pitch": 4.0,
        "wrist_yaw": 4.0,
    },
    dof_kd={
        "hip_yaw": 5.0,
        "hip_roll": 5.0,
        "hip_pitch": 5.0,
        "knee": 5.0,
        "ankle_pitch": 0.2,
        "ankle_roll": 0.1,
        "waist_yaw": 5.0,
        "waist_roll": 5.0,
        "waist_pitch": 5.0,
        "shoulder_pitch": 2.0,
        "shoulder_roll": 1.0,
        "shoulder_yaw": 0.4,
        "elbow": 1.0,
        "wrist_roll": 0.2,
        "wrist_pitch": 0.2,
        "wrist_yaw": 0.2,
    },
    action_scale=0.25,
    ctrl_freq=50,
    decimation=4,
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
