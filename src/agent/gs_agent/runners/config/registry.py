from pathlib import Path

from gs_agent.runners.config.schema import RunnerArgs

RUNNER_DEFAULT = RunnerArgs(
    total_iterations=100,
    log_interval=10,
    save_interval=100,
    save_path=Path("./logs/default"),
)


RUNNER_PENDULUM_PPO_MLP = RunnerArgs(
    total_iterations=500,
    log_interval=10,
    save_interval=100,
    save_path=Path("./logs/ppo_gym_pendulum"),
)

RUNNER_PENDULUM_BC_MLP = RunnerArgs(
    total_iterations=500,
    log_interval=10,
    save_interval=100,
    save_path=Path("./logs/bc_gym_pendulum"),
)

RUNNER_GOAL_REACHING_MLP = RunnerArgs(
    total_iterations=500,
    log_interval=10,
    save_interval=100,
    save_path=Path("./logs/ppo_gs_goal_reaching"),
)


RUNNER_WALKING_MLP = RunnerArgs(
    total_iterations=2000,
    log_interval=50,
    save_interval=400,
    save_path=Path("./logs/ppo_gs_walking"),
)
