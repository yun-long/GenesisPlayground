from gs_agent.runners.config.schema import RunnerArgs
from pathlib import Path



RUNNER_DEFAULT = RunnerArgs(
    total_iterations=100,
    log_interval=10,
    save_interval=100,
    save_path=Path("./logs/default"),
)


RUNNER_PENDULUM_MLP = RunnerArgs(
    total_iterations=500,
    log_interval=10,
    save_interval=100,
    save_path=Path("./logs/ppo_gym_pendulum"),
)

RUNNER_GOAL_REACHING_MLP = RunnerArgs(
    total_iterations=500,
    log_interval=10,
    save_interval=100,
    save_path=Path("./logs/ppo_gs_goal_reaching"),
)
