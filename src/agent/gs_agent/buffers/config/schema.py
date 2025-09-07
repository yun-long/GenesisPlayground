from enum import Enum

from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel


class GAEBufferKey(str, Enum):
    ACTOR_OBS = "ACTOR_OBS"
    ACTIONS = "ACTIONS"
    REWARDS = "REWARDS"
    DONES = "DONES"
    VALUES = "VALUES"
    ACTION_LOGPROBS = "ACTION_LOGPROBS"
    ADVANTAGES = "ADVANTAGES"
    RETURNS = "RETURNS"


class BCBufferKey(str, Enum):
    OBSERVATIONS = "OBSERVATIONS"
    ACTIONS = "ACTIONS"
    REWARDS = "REWARDS"
    DONES = "DONES"
    NEXT_OBSERVATIONS = "NEXT_OBSERVATIONS"


class GAEBufferArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    pass


class BCBufferArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    pass
