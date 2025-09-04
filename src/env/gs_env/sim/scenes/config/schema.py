from typing import TypeAlias

from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel

import genesis as gs


class BaseSceneArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    center_envs_at_origin: bool
    compile_kernels: bool

    sim_options: gs.options.SimOptions
    tool_options: gs.options.ToolOptions
    rigid_options: gs.options.RigidOptions
    avatar_options: gs.options.AvatarOptions
    mpm_options: gs.options.MPMOptions
    # sph_options: gs.options.SPHOptions
    fem_options: gs.options.FEMOptions
    sf_options: gs.options.SFOptions
    # pbd_options: gs.options.PBDOptions
    vis_options: gs.options.VisOptions
    viewer_options: gs.options.ViewerOptions

    show_viewer: bool
    show_FPS: bool


class FlatSceneArgs(BaseSceneArgs):
    model_config = genesis_pydantic_config(frozen=True)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)


SceneArgs: TypeAlias = FlatSceneArgs
