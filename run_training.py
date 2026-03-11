"""Run all three SC-MAPPO training phases sequentially.

Each phase saves a final checkpoint that is automatically passed to the next.
If a phase fails or you interrupt, re-run this script — it will find the most
recent checkpoint and skip completed phases.

Usage:
    uv run python run_training.py
    uv run python run_training.py --phase1-steps 200000   # quick smoke test
    uv run python run_training.py --start-phase 2 --checkpoint runs/phase1_sc_mappo__1__XYZ/checkpoints/ckpt_final.pt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_latest_checkpoint(exp_name: str) -> Path | None:
    """Return the ckpt_final.pt from the most recently modified run directory."""
    runs = sorted(
        Path("runs").glob(f"{exp_name}__*/checkpoints/ckpt_final.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


def run_phase(phase: int, extra_args: list[str]) -> Path:
    """Run one training phase. Returns the path to the saved final checkpoint."""
    exp_name = f"phase{phase}_sc_mappo"
    cmd = [
        sys.executable, "-m", "uv", "run", "python", "train.py",
        "--phase", str(phase),
        "--exp-name", exp_name,
        *extra_args,
    ]
    # Use uv run directly rather than going through the module system
    cmd = ["uv", "run", "python", "train.py",
           "--phase", str(phase),
           "--exp-name", exp_name,
           *extra_args]

    print(f"\n{'=' * 60}")
    print(f"  Starting Phase {phase}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Phase {phase} failed with exit code {result.returncode}")

    ckpt = find_latest_checkpoint(exp_name)
    if ckpt is None:
        sys.exit(f"Phase {phase} finished but no checkpoint found under runs/{exp_name}__*/")
    print(f"\n[run_training] Phase {phase} complete. Checkpoint: {ckpt}")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all SC-MAPPO training phases.")
    parser.add_argument("--start-phase", type=int, default=1, choices=[1, 2, 3],
                        help="Phase to start from (default: 1).")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Checkpoint to load when starting from phase 2 or 3.")

    # Pass-through hyperparameter overrides
    parser.add_argument("--phase1-steps", type=int, default=500_000)
    parser.add_argument("--phase2-steps", type=int, default=3_000_000)
    parser.add_argument("--phase3-steps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--n-robots", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    shared_args = [
        "--num-envs", str(args.num_envs),
        "--n-robots", str(args.n_robots),
        "--seed",     str(args.seed),
    ]

    ckpt_path = args.checkpoint

    # ----------------------------------------------------------------
    # Phase 1 — Locomotion bootstrap
    # ----------------------------------------------------------------
    if args.start_phase <= 1:
        ckpt_path = str(run_phase(1, [
            "--total-timesteps", str(args.phase1_steps),
            "--phase1-steps",    str(args.phase1_steps),
            *shared_args,
        ]))

    # ----------------------------------------------------------------
    # Phase 2 — Multi-task coordination
    # ----------------------------------------------------------------
    if args.start_phase <= 2:
        if not ckpt_path:
            sys.exit("--checkpoint required when starting from phase 2")
        ckpt_path = str(run_phase(2, [
            "--total-timesteps", str(args.phase2_steps),
            "--checkpoint",      ckpt_path,
            *shared_args,
        ]))

    # ----------------------------------------------------------------
    # Phase 3 — End-to-end fine-tuning
    # ----------------------------------------------------------------
    if args.start_phase <= 3:
        if not ckpt_path:
            sys.exit("--checkpoint required when starting from phase 3")
        run_phase(3, [
            "--total-timesteps", str(args.phase3_steps),
            "--checkpoint",      ckpt_path,
            "--learning-rate",   "3e-5",
            *shared_args,
        ])

    print("\n[run_training] All phases complete.")
    print("  Evaluate:   uv run python evaluate.py --checkpoint runs/phase3_sc_mappo__*/checkpoints/ckpt_final.pt")
    print("  TensorBoard: tensorboard --logdir runs/")


if __name__ == "__main__":
    main()
