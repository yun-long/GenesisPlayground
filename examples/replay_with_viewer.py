#!/usr/bin/env python3
"""
Replay trajectory with proper viewer support.

- Forces Genesis to CPU backend to avoid MPS/NumPy conversion errors on macOS.
- Lets you choose a specific trajectory via CLI.
- Keeps the viewer active by stepping the sim between commands.

Usage:
  python examples/replay_so101_trajectory.py                 # uses newest trajectory
  python examples/replay_so101_trajectory.py --file path/to/so101_pick_place_YYYYmmdd_HHMMSS.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
from typing import List

# -----------------------------------------------------------------------------
# Add the repo's src directory to sys.path so imports work when run from examples/
# Adjust if your layout differs.
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Project imports (update these paths if your package names differ)
from agent.gs_agent.wrappers.teleop_wrapper import TeleopWrapper, TeleopCommand  # type: ignore
from env.gs_env.sim.envs.so101_cube_env import SO101CubeEnv  # type: ignore

import numpy as np
import genesis as gs


def list_trajectories(dirpath: str = "trajectories") -> List[str]:
    """Return all trajectory files, newest first."""
    if not os.path.isdir(dirpath):
        return []
    files = [
        os.path.join(dirpath, f)
        for f in os.listdir(dirpath)
        if f.startswith("so101_pick_place_") and f.endswith(".json")
    ]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def pick_trajectory(path_arg: str | None) -> str | None:
    """Pick a trajectory file: use arg if given, otherwise newest available."""
    if path_arg:
        if os.path.isfile(path_arg):
            return path_arg
        print(f"❌ Trajectory file not found: {path_arg}")
        return None

    files = list_trajectories()
    if not files:
        print("❌ No trajectory files found in ./trajectories")
        print("   Record one first by running your teleop and pressing 'r' to start/stop recording.")
        return None

    print(f"📁 Found {len(files)} trajectory file(s):")
    for i, fp in enumerate(files, 1):
        print(f"  {i}. {os.path.basename(fp)}")
    print(f"👉 Using most recent: {os.path.basename(files[0])}")
    return files[0]


def init_genesis_cpu() -> None:
    """Initialize Genesis on CPU backend to keep tensors on host memory."""
    gs.init(
        seed=0,
        precision="32",
        logging_level="info",
        backend=gs.cpu  # type: ignore  # IMPORTANT: avoid MPS/GPU tensors during replay
    )
    print("✅ Genesis initialized (CPU backend)")


def replay_trajectory(traj_path: str) -> None:
    """Replay a saved trajectory with viewer updates."""
    # Load trajectory JSON
    with open(traj_path, "r") as f:
        trajectory_info = json.load(f)

    trajectory = trajectory_info.get("trajectory", [])
    meta = trajectory_info.get("metadata", {})

    print(f"\n🎬 Replaying: {os.path.basename(traj_path)}")
    print(f"   - Steps: {len(trajectory)}")
    print(f"   - Duration: {meta.get('duration_seconds', 0):.2f}s")
    print(f"   - Task: {meta.get('task', 'unknown')}\n")

    # Build env + wrapper (viewer owned by env)
    teleop_wrapper = TeleopWrapper()
    env = SO101CubeEnv()
    teleop_wrapper.set_environment(env)

    # Reset env to a clean state
    print("🔄 Resetting environment...")
    env.reset_idx(None)

    # Small pause so the viewer is fully ready
    print("⏳ Waiting for viewer to be ready...")
    time.sleep(2.0)

    # Replay loop
    print("▶️  Starting replay (Ctrl+C to stop)")
    try:
        # Tune these for smoothness vs. speed
        inner_steps_per_cmd = 12   # sim steps per command
        dt_between_steps = 0.015   # seconds per sim step

        for i, step in enumerate(trajectory):
            cmd = step.get("command", {})
            command = TeleopCommand(
                position=np.array(cmd.get("position", [0.0, 0.0, 0.0]), dtype=float),
                orientation=np.array(cmd.get("orientation", [0.0, 0.0, 0.0]), dtype=float),
                gripper_close=bool(cmd.get("gripper_close", False)),
                reset_scene=bool(cmd.get("reset_scene", False)),
                quit_teleop=bool(cmd.get("quit_teleop", False)),
            )

            # Apply the recorded command
            env.apply_command(command)

            # Step multiple times to keep the viewer active/smooth
            for _ in range(inner_steps_per_cmd):
                env.step()
                time.sleep(dt_between_steps)

            # Progress log every ~10%
            if (i + 1) % max(1, len(trajectory) // 10) == 0:
                progress = (i + 1) / max(1, len(trajectory)) * 100.0
                print(f"   Progress: {progress:.0f}% ({i + 1}/{len(trajectory)})")

        print("✅ Trajectory replay completed!")
        print("👀 Keeping viewer open for a few seconds...")
        time.sleep(3.0)

    except KeyboardInterrupt:
        print("\n⏹️  Replay stopped by user.")
    except Exception as e:
        print(f"❌ Failed during replay: {e}")
    finally:
        print("🧹 Cleaning up...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay SO101 teleop trajectory with viewer.")
    parser.add_argument("--file", "-f", type=str, default=None, help="Path to a trajectory JSON file.")
    args = parser.parse_args()

    print("Initializing SO101 Replay System with Viewer…")
    init_genesis_cpu()

    traj_path = pick_trajectory(args.file)
    if not traj_path:
        return

    replay_trajectory(traj_path)


if __name__ == "__main__":
    main()
