from pathlib import Path

from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel, NonNegativeInt

# ============================================================================
# Runner Configuration
# ============================================================================


class RunnerArgs(BaseModel):
    """Configuration for on-policy runners."""

    model_config = genesis_pydantic_config(frozen=True)

    total_iterations: NonNegativeInt = 1000

    # Training intervals
    log_interval: NonNegativeInt = 10
    save_interval: NonNegativeInt = 100

    save_path: Path = Path(".")
