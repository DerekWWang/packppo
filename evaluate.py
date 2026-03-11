"""Evaluate a trained SC-MAPPO checkpoint with optional MuJoCo viewer.

Usage examples:
    # Render task 0 with 2 robots
    uv run python evaluate.py --checkpoint runs/sc_mappo__1__1234/checkpoints/ckpt_final.pt

    # Evaluate task 2 (v_formation) with 3 robots, no rendering
    uv run python evaluate.py --checkpoint ... --task-id 2 --n-robots 3 --num-episodes 20

    # Render all 4 tasks in sequence
    uv run python evaluate.py --checkpoint ... --all-tasks
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from envs.make_env import make_env, TASK_NAMES
from envs.multi_ant_base import PROPRIO_DIM, ROBOT_ACTUATORS
from networks.agent import SCMAPPOAgent


@dataclass
class EvalArgs:
    checkpoint: str
    """Path to a saved checkpoint .pt file."""
    task_id: int = 0
    """Task to evaluate (0–3). Ignored if --all-tasks."""
    n_robots: int = 2
    """Number of robots (2 or 3)."""
    num_episodes: int = 5
    """Number of evaluation episodes to run."""
    task_emb_dim: int = 8
    n_tasks: int = 4
    attn_hidden: int = 64
    critic_hidden: int = 256
    render: bool = True
    """Open a MuJoCo viewer window."""
    render_fps: int = 50
    """Target rendering frame rate."""
    all_tasks: bool = False
    """Evaluate all 4 tasks sequentially."""
    deterministic: bool = True
    """Use deterministic (greedy) actions."""
    cuda: bool = False
    """Use CUDA for inference (usually not needed for evaluation)."""
    include_cfrc_ext: bool | None = None
    """Include raw contact-force obs (78 dims). None = auto-detect from checkpoint."""


def evaluate_task(
    agent: SCMAPPOAgent,
    task_id: int,
    n_robots: int,
    num_episodes: int,
    task_emb_dim: int,
    render: bool,
    render_fps: int,
    deterministic: bool,
    device: torch.device,
    include_cfrc_ext: bool = False,
) -> dict[str, float]:
    """Run num_episodes of a task and return aggregate statistics."""
    render_mode = "human" if render else None
    env = make_env(
        task_id=task_id,
        n_robots=n_robots,
        seed=0,
        task_embedding_dim=task_emb_dim,
        render_mode=render_mode,
        include_cfrc_ext=include_cfrc_ext,
    )()

    obs_dim = env.obs_dim_per_robot
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    viewer_ctx = None

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()

        # Open viewer on the first episode (after reset so the scene is populated)
        if render and viewer_ctx is None:
            try:
                import mujoco.viewer
                viewer_ctx = mujoco.viewer.launch_passive(env.model, env.data)
            except Exception as e:
                print(f"  [warn] Could not open MuJoCo viewer: {e}")
                render = False
        obs_np = obs_dict["obs"]  # (N, obs_dim)

        # Get task embedding from agent
        with torch.no_grad():
            task_id_t = torch.tensor([task_id], device=device, dtype=torch.long)
            task_emb_np = agent.task_embeddings(task_id_t).squeeze(0).cpu().numpy()
        env._task_emb = task_emb_np

        ep_return = 0.0
        ep_length = 0
        done = False

        while not done:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
            obs_flat = obs_t.unsqueeze(0).reshape(n_robots, obs_dim)  # (N, obs_dim)

            with torch.no_grad():
                task_id_batch = torch.tensor([task_id], device=device, dtype=torch.long)
                actions, _, _, _ = agent.get_action_and_value(
                    obs_flat, task_id_batch, n_robots=n_robots,
                    deterministic=deterministic
                )

            actions_np = actions.cpu().numpy().reshape(n_robots, ROBOT_ACTUATORS)
            obs_dict, reward, terminated, truncated, info = env.step(actions_np)
            obs_np = obs_dict["obs"]
            env._task_emb = task_emb_np

            ep_return += reward
            ep_length += 1
            done = terminated or truncated

            if render and viewer_ctx is not None:
                viewer_ctx.sync()
                time.sleep(1.0 / render_fps)

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        print(
            f"  Episode {ep + 1}/{num_episodes}: "
            f"return={ep_return:.2f}, length={ep_length}"
        )

    if viewer_ctx is not None:
        input("  [viewer] All episodes done — press Enter to close the viewer...")
        viewer_ctx.close()
    env.close()

    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return":  float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
    }


def main() -> None:
    args = tyro.cli(EvalArgs)

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Auto-detect include_cfrc_ext from saved training args if not overridden
    if args.include_cfrc_ext is None:
        include_cfrc_ext: bool = ckpt.get("args", {}).get("include_cfrc_ext", False)
    else:
        include_cfrc_ext = args.include_cfrc_ext

    # Build agent (infer obs_dim from a temp env)
    tmp_env = make_env(
        task_id=0, n_robots=args.n_robots, seed=999,
        task_embedding_dim=args.task_emb_dim,
        include_cfrc_ext=include_cfrc_ext,
    )()
    obs_dim = tmp_env.obs_dim_per_robot
    tmp_env.close()

    agent = SCMAPPOAgent(
        obs_dim_per_robot=obs_dim,
        proprio_dim=tmp_env.proprio_dim,
        action_dim=ROBOT_ACTUATORS,
        n_robots=args.n_robots,
        n_tasks=args.n_tasks,
        task_emb_dim=args.task_emb_dim,
        attn_hidden=args.attn_hidden,
        critic_hidden=args.critic_hidden,
    ).to(device)

    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    # Activate attention if it was enabled during training
    saved_phase = ckpt.get("current_phase", 1)
    if saved_phase >= 2:
        agent.enable_attention()
    print(
        f"[evaluate] Loaded checkpoint (phase={saved_phase}, "
        f"step={ckpt.get('global_step', '?'):,})"
    )

    # Determine which tasks to evaluate
    tasks_to_run = list(range(args.n_tasks)) if args.all_tasks else [args.task_id]

    for task_id in tasks_to_run:
        print(f"\n=== Task {task_id}: {TASK_NAMES[task_id]} ===")
        stats = evaluate_task(
            agent=agent,
            task_id=task_id,
            n_robots=args.n_robots,
            num_episodes=args.num_episodes,
            task_emb_dim=args.task_emb_dim,
            render=args.render,
            render_fps=args.render_fps,
            deterministic=args.deterministic,
            device=device,
            include_cfrc_ext=include_cfrc_ext,
        )
        print(
            f"  Summary: mean_return={stats['mean_return']:.2f} "
            f"(±{stats['std_return']:.2f}), "
            f"mean_length={stats['mean_length']:.1f}"
        )


if __name__ == "__main__":
    main()
