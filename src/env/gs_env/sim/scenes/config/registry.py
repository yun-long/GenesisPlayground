import genesis as gs

from gs_env.sim.scenes.config.schema import (
    FlatSceneArgs,
    SceneArgs,
)

# NOTE: break dependencies on default values from Genesis to avoid silent bugs
# due to any changes in the Genesis repo.


# ------------------------------------------------------------
# Sim
# ------------------------------------------------------------

SimArgsRegistry: dict[str, gs.options.SimOptions] = {}

SimArgsRegistry["default"] = gs.options.SimOptions(
    dt=1e-2,
    substeps=1,
    substeps_local=None,
    gravity=(0.0, 0.0, -9.81),
    floor_height=0.0,
    requires_grad=False,
)


# ------------------------------------------------------------
# Tool
# ------------------------------------------------------------

ToolArgsRegistry: dict[str, gs.options.ToolOptions] = {}

ToolArgsRegistry["default"] = gs.options.ToolOptions(
    dt=None,
    floor_height=0.0,
)


# ------------------------------------------------------------
# Rigid
# ------------------------------------------------------------

RigidArgsRegistry: dict[str, gs.options.RigidOptions] = {}

RigidArgsRegistry["default"] = gs.options.RigidOptions(
    dt=None,
    gravity=None,
    enable_collision=True,
    enable_joint_limit=True,
    enable_self_collision=False,
    enable_adjacent_collision=False,
    max_collision_pairs=100,
    integrator=gs.integrator.approximate_implicitfast,
    IK_max_targets=6,
    constraint_solver=gs.constraint_solver.CG,
    iterations=100,
    tolerance=1e-5,
    ls_iterations=50,
    ls_tolerance=1e-2,
    sparse_solve=False,
    contact_resolve_time=None,
    use_contact_island=False,
    use_hibernation=False,
    hibernation_thresh_vel=1e-3,
    hibernation_thresh_acc=1e-2,
    box_box_detection=False,
)


# ------------------------------------------------------------
# Avatar
# ------------------------------------------------------------

AvatarArgsRegistry: dict[str, gs.options.AvatarOptions] = {}

AvatarArgsRegistry["default"] = gs.options.AvatarOptions(
    dt=None,
    enable_collision=False,
    enable_self_collision=False,
    enable_adjacent_collision=False,
    max_collision_pairs=100,
    IK_max_targets=6,
)


# ------------------------------------------------------------
# MPM
# ------------------------------------------------------------

MPMArgsRegistry: dict[str, gs.options.MPMOptions] = {}

MPMArgsRegistry["default"] = gs.options.MPMOptions(
    dt=None,
    gravity=None,
    particle_size=None,
    grid_density=64,
    enable_CPIC=False,
    lower_bound=(-1.0, -1.0, 0.0),
    upper_bound=(1.0, 1.0, 1.0),
    use_sparse_grid=False,
    leaf_block_size=8,
)


# ------------------------------------------------------------
# SPH
# ------------------------------------------------------------

# SPHArgsRegistry: dict[str, gs.options.SPHOptions] = {}
#
# SPHArgsRegistry["default"] = gs.options.SPHOptions(
#     dt=None,
#     gravity=None,
#     particle_size=0.02,
#     pressure_solver="WCSPH",
#     lower_bound=(-100.0, -100.0, 0.0),
#     upper_bound=(100.0, 100.0, 100.0),
#     hash_grid_res=None,
#     hash_grid_cell_size=None,
#     max_divergence_error=0.1,
#     max_density_error_percent=0.05,
#     max_divergence_solver_iterations=100,
#     max_density_solver_iterations=100,
# )


# ------------------------------------------------------------
# FEM
# ------------------------------------------------------------

FEMArgsRegistry: dict[str, gs.options.FEMOptions] = {}

FEMArgsRegistry["default"] = gs.options.FEMOptions(
    dt=None,
    gravity=None,
    damping=0.0,
    floor_height=0.0,
)


# ------------------------------------------------------------
# SF
# ------------------------------------------------------------

SFArgsRegistry: dict[str, gs.options.SFOptions] = {}

SFArgsRegistry["default"] = gs.options.SFOptions(
    dt=None,
    res=128,
    solver_iters=500,
    decay=0.99,
    T_low=1.0,
    T_high=0.0,
    inlet_pos=(0, 0, 0),
    inlet_vel=(0, 0, 1),
    inlet_quat=(1, 0, 0, 0),
    inlet_s=400.0,
)


# ------------------------------------------------------------
# PBD
# ------------------------------------------------------------
#
# PBDArgsRegistry: dict[str, gs.options.PBDOptions] = {}
#
# PBDArgsRegistry["default"] = gs.options.PBDOptions(
#     dt=None,
#     gravity=None,
#     max_stretch_solver_iterations=4,
#     max_bending_solver_iterations=1,
#     max_volume_solver_iterations=1,
#     max_density_solver_iterations=1,
#     max_viscosity_solver_iterations=1,
#     particle_size=1e-2,
#     hash_grid_res=None,
#     hash_grid_cell_size=None,
#     lower_bound=(-100.0, -100.0, 0.0),
#     upper_bound=(100.0, 100.0, 100.0),
# )


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

VisArgsRegistry: dict[str, gs.options.VisOptions] = {}

VisArgsRegistry["default"] = gs.options.VisOptions(
    show_world_frame=True,
    world_frame_size=1.0,
    show_link_frame=False,
    link_frame_size=0.2,
    show_cameras=False,
    shadow=True,
    plane_reflection=False,
    env_separate_rigid=False,
    background_color=(0.04, 0.08, 0.12),
    ambient_light=(0.1, 0.1, 0.1),
    visualize_mpm_boundary=False,
    visualize_sph_boundary=False,
    visualize_pbd_boundary=False,
    segmentation_level="link",
    render_particle_as="sphere",
    particle_size_scale=1.0,
    contact_force_scale=0.01,
    n_support_neighbors=12,
    n_rendered_envs=None,
    lights=[
        {"type": "directional", "dir": (-1, -1, -1), "color": (1.0, 1.0, 1.0), "intensity": 5.0},
    ],
)


# ------------------------------------------------------------
# Viewer
# ------------------------------------------------------------

ViewerArgsRegistry: dict[str, gs.options.ViewerOptions] = {}

ViewerArgsRegistry["default"] = gs.options.ViewerOptions(
    res=None,
    refresh_rate=60,
    max_FPS=60,
    camera_pos=(3.5, 0.5, 2.5),
    camera_lookat=(0.0, 0.0, 0.5),
    camera_up=(0.0, 0.0, 1.0),
    camera_fov=40,
)


# ------------------------------------------------------------
# Scene
# ------------------------------------------------------------


SceneArgsRegistry: dict[str, SceneArgs] = {}


SceneArgsRegistry["flat_scene_default"] = FlatSceneArgs(
    show_viewer=False,
    show_FPS=False,
    center_envs_at_origin=True,
    compile_kernels=True,
    sim_options=SimArgsRegistry["default"],
    tool_options=ToolArgsRegistry["default"],
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit=True,
        enable_collision=True,
        gravity=(0, 0, -9.8),
        box_box_detection=True,
    ),
    avatar_options=AvatarArgsRegistry["default"],
    mpm_options=MPMArgsRegistry["default"],
    # sph_options=SPHArgsRegistry["default"],
    fem_options=FEMArgsRegistry["default"],
    sf_options=SFArgsRegistry["default"],
    # pbd_options=PBDArgsRegistry["default"],
    vis_options=VisArgsRegistry["default"],
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(-0.6, 0.0, 0.7),
        camera_lookat=(0.2, 0.0, 0.1),
        camera_fov=50,
        max_FPS=60,
    ),
    normal=(0.0, 0.0, 1.0),
)
