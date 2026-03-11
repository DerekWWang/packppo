"""SC-MAPPO: Skill-Conditioned Multi-Agent PPO  (CleanRL-style single-file trainer)

Three-phase training:
  Phase 1  — Locomotion bootstrap (~5 min):
      All envs run follow-the-leader (task 0).
      Standard PPO, no attention, single value head.
      Warm-starts the locomotion backbone.

  Phase 2  — Multi-task coordination (~30-60 min):
      Freeze the first backbone layer.
      Enable cross-robot attention.
      All 4 tasks sampled via critic-uncertainty curriculum.

  Phase 3  — End-to-end fine-tuning (~30-60 min):
      Unfreeze all layers.
      Reduced learning rate (lr / 10).
      Curriculum continues.

Run examples:
    uv run python train.py                             # start from Phase 1
    uv run python train.py --phase 2 --checkpoint runs/phase1/ckpt.pt
    uv run python train.py --phase 3 --checkpoint runs/phase2/ckpt.pt --learning-rate 3e-5

Monitor:
    tensorboard --logdir runs/
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from envs.make_env import make_env, TASK_NAMES
from envs.multi_ant_base import PROPRIO_DIM, ROBOT_ACTUATORS
from networks.agent import SCMAPPOAgent
from utils.curriculum import CurriculumSampler
from utils.popart import PopArtNormalizer


# ---------------------------------------------------------------------------
# Hyperparameters (all configurable via CLI)
# ---------------------------------------------------------------------------

@dataclass
class Args:
    # Experiment
    exp_name: str = "sc_mappo"
    """Name for the TensorBoard run directory."""
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    use_amp: bool = True
    """Use automatic mixed precision (fp16) for the PPO update.  Requires CUDA.
    Gives ~1.5-2x speedup on modern GPUs with no loss in training quality."""

    # Environment
    n_robots: int = 2
    """Number of robots per environment (2 or 3)."""
    n_tasks: int = 4
    task_emb_dim: int = 8
    """Dimension of the learned task embedding vector."""

    # Rollout
    total_timesteps: int = 10_000_000
    num_envs: int = 16
    """Number of parallel environments. More envs = better GPU utilisation
    during the PPO update; diminishing returns beyond ~32 for this model size."""
    num_steps: int = 128
    """Rollout length per environment per update."""
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO update
    learning_rate: float = 3e-4
    num_epochs: int = 10
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    """Normalize advantages per minibatch."""
    target_kl: float | None = None
    """Early stop PPO epochs when approx KL exceeds this value (optional)."""

    # SC-MAPPO specific
    attn_hidden: int = 64
    critic_hidden: int = 256
    curriculum_temp: float = 1.0
    curriculum_window: int = 100
    """Rolling window length for TD-error variance estimates."""

    # Observation
    include_cfrc_ext: bool = False
    """Include raw contact-force observations (78 dims, unscaled).  Disabled by
    default: cfrc_ext values are orders of magnitude larger than joint angles /
    velocities and hurt early locomotion learning.  Enable only if you need
    contact-force sensing and add obs normalisation."""

    # Phase control
    phase: int = 1
    """Starting phase: 1, 2, or 3."""
    phase1_steps: int = 5_000_000
    """Total env-steps for Phase 1 before auto-advancing to Phase 2.
    Quadruped locomotion from scratch typically requires ~5 M steps."""
    phase2_steps: int = 3_000_000
    """Total env-steps for Phase 2 before auto-advancing to Phase 3."""
    checkpoint: str = ""
    """Path to a checkpoint .pt file to resume from."""
    save_interval: int = 100
    """Save a checkpoint every N PPO updates."""

    # Logging
    track: bool = False
    """Log to Weights & Biases in addition to TensorBoard."""
    wandb_project: str = "sc_mappo"
    wandb_entity: str = ""


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBatch(NamedTuple):
    obs: torch.Tensor          # (T, num_envs, N, obs_dim)
    actions: torch.Tensor      # (T, num_envs, N, action_dim)
    log_probs: torch.Tensor    # (T, num_envs, N)
    rewards: torch.Tensor      # (T, num_envs)
    dones: torch.Tensor        # (T, num_envs)
    values: torch.Tensor       # (T, num_envs)
    task_ids: torch.Tensor     # (T, num_envs)  long


def compute_gae(
    rewards: torch.Tensor,   # (T, num_envs)
    values: torch.Tensor,    # (T, num_envs)
    next_value: torch.Tensor,  # (num_envs,)
    dones: torch.Tensor,     # (T, num_envs)
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    Returns:
        advantages: (T, num_envs)
        returns:    (T, num_envs)
    """
    T, E = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(E, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
            next_done = torch.zeros(E, device=rewards.device)
        else:
            next_val = values[t + 1]
            next_done = dones[t + 1]

        delta = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - next_done) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Phase transition helpers
# ---------------------------------------------------------------------------

def transition_to_phase2(
    agent: SCMAPPOAgent,
    optimizer: optim.Optimizer,
    args: Args,
) -> None:
    """Freeze first backbone layer, enable attention."""
    print("\n[SC-MAPPO] Advancing to Phase 2: multi-task coordination")
    agent.enable_attention()
    agent.freeze_backbone_lower()
    # Rebuild optimizer param groups (frozen params excluded)
    optimizer.param_groups.clear()
    optimizer.add_param_group({"params": [p for p in agent.parameters() if p.requires_grad]})


def transition_to_phase3(
    agent: SCMAPPOAgent,
    optimizer: optim.Optimizer,
    args: Args,
) -> None:
    """Unfreeze all layers, reduce LR by 10×."""
    print("\n[SC-MAPPO] Advancing to Phase 3: end-to-end fine-tuning")
    agent.unfreeze_all()
    new_lr = args.learning_rate / 10.0
    optimizer.param_groups.clear()
    optimizer.add_param_group({
        "params": list(agent.parameters()),
        "lr": new_lr,
    })
    print(f"  Learning rate reduced to {new_lr:.2e}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    agent: SCMAPPOAgent,
    optimizer: optim.Optimizer,
    global_step: int,
    current_phase: int,
    args: Args,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "current_phase": current_phase,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    agent: SCMAPPOAgent,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    """Load checkpoint. Returns (global_step, current_phase)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[SC-MAPPO] Loaded checkpoint from {path}")
    print(f"  global_step={ckpt['global_step']}, phase={ckpt['current_phase']}")
    return ckpt["global_step"], ckpt["current_phase"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    args = tyro.cli(Args)

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    # Enable cuDNN auto-tuner when not in strict deterministic mode.
    # Finds the fastest conv algorithm for fixed input sizes — free throughput.
    if device.type == "cuda" and not args.torch_deterministic:
        torch.backends.cudnn.benchmark = True

    # AMP (automatic mixed precision): fp16 forward + fp32 param updates.
    # Only active when CUDA is available.
    use_amp = args.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[SC-MAPPO] Device: {device}  ({gpu_name}, {gpu_mem:.1f} GB)")
        print(f"  AMP (mixed precision): {'ON' if use_amp else 'OFF'}")
        print(f"  cudnn.benchmark:       {torch.backends.cudnn.benchmark}")
    else:
        print(f"[SC-MAPPO] Device: {device}  (no CUDA — AMP disabled)")

    # TensorBoard
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    log_dir = Path("runs") / run_name
    writer = SummaryWriter(str(log_dir))
    writer.add_text("hyperparameters", str(vars(args)))

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            config=vars(args),
            name=run_name,
            sync_tensorboard=True,
        )

    # ------------------------------------------------------------------
    # Build environments (manual list — SyncVectorEnv doesn't handle
    # dict obs with (N, obs_dim) arrays cleanly)
    # ------------------------------------------------------------------
    # Phase 1: all envs run task 0 (follow-the-leader = simplest locomotion)
    envs = [
        make_env(task_id=0, n_robots=args.n_robots, seed=args.seed + i,
                 task_embedding_dim=args.task_emb_dim,
                 include_cfrc_ext=args.include_cfrc_ext)()
        for i in range(args.num_envs)
    ]
    obs_dim = envs[0].obs_dim_per_robot
    proprio_dim = envs[0].proprio_dim
    action_dim = args.n_robots * ROBOT_ACTUATORS  # unused as scalar; per-robot is ROBOT_ACTUATORS

    # ------------------------------------------------------------------
    # Build agent
    # ------------------------------------------------------------------
    agent = SCMAPPOAgent(
        obs_dim_per_robot=obs_dim,
        proprio_dim=proprio_dim,
        action_dim=ROBOT_ACTUATORS,
        n_robots=args.n_robots,
        n_tasks=args.n_tasks,
        task_emb_dim=args.task_emb_dim,
        attn_hidden=args.attn_hidden,
        critic_hidden=args.critic_hidden,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # PopArt normalizer (per-task value target normalization)
    popart = PopArtNormalizer(n_tasks=args.n_tasks).to(device)

    # Curriculum sampler
    curriculum = CurriculumSampler(
        n_tasks=args.n_tasks,
        window=args.curriculum_window,
        temperature=args.curriculum_temp,
    )

    # ------------------------------------------------------------------
    # Load checkpoint if provided
    # ------------------------------------------------------------------
    global_step = 0
    current_phase = args.phase

    if args.checkpoint:
        global_step, loaded_phase = load_checkpoint(
            args.checkpoint, agent, optimizer, device
        )
        # Respect the starting phase from CLI (allows manual override)
        if loaded_phase != args.phase:
            print(
                f"  Note: checkpoint phase={loaded_phase}, "
                f"CLI --phase={args.phase} (using CLI value)"
            )

    # Apply phase-specific setup for non-Phase-1 starts
    if current_phase >= 2:
        agent.enable_attention()
        agent.freeze_backbone_lower()
    if current_phase == 3:
        agent.unfreeze_all()

    # ------------------------------------------------------------------
    # Rollout buffers (pre-allocated on device)
    # ------------------------------------------------------------------
    T, E, N = args.num_steps, args.num_envs, args.n_robots
    buf_obs      = torch.zeros((T, E, N, obs_dim), device=device)
    buf_actions  = torch.zeros((T, E, N, ROBOT_ACTUATORS), device=device)
    buf_logprobs = torch.zeros((T, E, N), device=device)
    buf_rewards  = torch.zeros((T, E), device=device)
    buf_dones    = torch.zeros((T, E), device=device)
    buf_values   = torch.zeros((T, E), device=device)
    buf_taskids  = torch.zeros((T, E), dtype=torch.long, device=device)

    # Reset all environments and collect initial observations
    # task_emb_np will be updated each rollout from the agent's Embedding table
    next_obs_np  = np.stack([envs[i].reset()[0]["obs"] for i in range(E)])  # (E, N, obs_dim)
    next_done_np = np.zeros(E, dtype=np.float32)

    # Pinned-memory staging tensors for non-blocking CPU→GPU transfers.
    # .pin_memory() keeps the tensor in page-locked host RAM so the GPU DMA
    # engine can copy it asynchronously while the CPU runs env steps.
    if device.type == "cuda":
        _obs_staging  = torch.zeros((E, N, obs_dim), dtype=torch.float32).pin_memory()
        _done_staging = torch.zeros(E, dtype=torch.float32).pin_memory()
        _rew_staging  = torch.zeros(E, dtype=torch.float32).pin_memory()
    else:
        _obs_staging  = None
        _done_staging = None
        _rew_staging  = None

    def _to_gpu_obs(arr: np.ndarray) -> torch.Tensor:
        """Copy (E, N, obs_dim) numpy array to GPU with minimal blocking."""
        if _obs_staging is not None:
            _obs_staging.copy_(torch.from_numpy(arr))
            return _obs_staging.to(device, non_blocking=True)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def _to_gpu_done(arr: np.ndarray) -> torch.Tensor:
        if _done_staging is not None:
            _done_staging.copy_(torch.from_numpy(arr))
            return _done_staging.to(device, non_blocking=True)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def _to_gpu_rew(lst: list[float]) -> torch.Tensor:
        if _rew_staging is not None:
            _rew_staging.copy_(torch.tensor(lst, dtype=torch.float32))
            return _rew_staging.to(device, non_blocking=True)
        return torch.tensor(lst, dtype=torch.float32, device=device)

    # Choose initial task
    current_task_id = 0  # Phase 1 always task 0

    # Training statistics
    num_updates = args.total_timesteps // (T * E)
    batch_size = T * E
    minibatch_size = batch_size // args.num_minibatches

    print(
        f"[SC-MAPPO] Starting Phase {current_phase}, "
        f"total_timesteps={args.total_timesteps:,}, "
        f"num_updates={num_updates:,}, "
        f"batch_size={batch_size:,}"
    )

    checkpoint_dir = log_dir / "checkpoints"

    start_time = time.time()

    for update in range(1, num_updates + 1):

        # ----------------------------------------------------------------
        # Auto-advance phases
        # ----------------------------------------------------------------
        if current_phase == 1 and global_step >= args.phase1_steps:
            current_phase = 2
            transition_to_phase2(agent, optimizer, args)
            # Switch envs to use curriculum
        if current_phase == 2 and global_step >= args.phase1_steps + args.phase2_steps:
            current_phase = 3
            transition_to_phase3(agent, optimizer, args)

        # ----------------------------------------------------------------
        # Sample task for this rollout
        # ----------------------------------------------------------------
        if current_phase == 1:
            current_task_id = 0
        else:
            current_task_id = curriculum.sample()

        # Re-create envs if task changed (cheapest correct approach)
        # We only swap envs if the task changed between rollouts
        # (avoids recreating every step while still supporting task switching)
        if current_phase > 1:
            for i, env in enumerate(envs):
                if env.task_id != current_task_id:
                    env.close()
                    envs[i] = make_env(
                        task_id=current_task_id,
                        n_robots=args.n_robots,
                        seed=args.seed + i + update * 1000,
                        task_embedding_dim=args.task_emb_dim,
                        include_cfrc_ext=args.include_cfrc_ext,
                    )()
                    next_obs_np[i] = envs[i].reset()[0]["obs"]
                    next_done_np[i] = 0.0

        # Write task embedding into all envs (so it appears in obs)
        with torch.no_grad():
            task_id_t = torch.tensor([current_task_id], device=device, dtype=torch.long)
            task_emb_np = (
                agent.task_embeddings(task_id_t).squeeze(0).cpu().numpy()
            )  # (task_emb_dim,)
        for env in envs:
            env._task_emb = task_emb_np

        # ----------------------------------------------------------------
        # Collect rollout
        # ----------------------------------------------------------------
        next_obs_t  = _to_gpu_obs(next_obs_np)
        next_done_t = _to_gpu_done(next_done_np)

        for step in range(T):
            buf_obs[step]     = next_obs_t
            buf_dones[step]   = next_done_t
            buf_taskids[step] = current_task_id

            with torch.no_grad():
                # Flatten to (E*N, obs_dim) for agent forward pass
                obs_flat = next_obs_t.reshape(E * N, obs_dim)
                task_id_batch = torch.full((E,), current_task_id,
                                           dtype=torch.long, device=device)

                actions, log_probs, _, values = agent.get_action_and_value(
                    obs_flat, task_id_batch, n_robots=N
                )

            buf_actions[step]  = actions.reshape(E, N, ROBOT_ACTUATORS)
            buf_logprobs[step] = log_probs.reshape(E, N)
            buf_values[step]   = values   # (E,)

            # Step all envs
            actions_np = actions.reshape(E, N, ROBOT_ACTUATORS).cpu().numpy()
            next_obs_list, rewards, terminateds, truncateds, infos = [], [], [], [], []
            for i, env in enumerate(envs):
                obs_i, rew_i, term_i, trunc_i, info_i = env.step(actions_np[i])
                next_obs_list.append(obs_i["obs"])
                rewards.append(rew_i)
                terminateds.append(term_i)
                truncateds.append(trunc_i)
                infos.append(info_i)

                # Log episode stats on termination
                if term_i or trunc_i:
                    ep_info = info_i.get("episode", {})
                    if ep_info:
                        writer.add_scalar(
                            f"charts/episodic_return_task{current_task_id}",
                            ep_info["r"], global_step
                        )
                        writer.add_scalar(
                            f"charts/episodic_length_task{current_task_id}",
                            ep_info["l"], global_step
                        )
                    # Auto-reset
                    reset_obs, _ = env.reset()
                    env._task_emb = task_emb_np
                    next_obs_list[-1] = reset_obs["obs"]

            next_obs_np  = np.stack(next_obs_list)                           # (E,N,obs_dim)
            next_done_np = np.array(
                [float(t or u) for t, u in zip(terminateds, truncateds)],
                dtype=np.float32
            )
            buf_rewards[step] = _to_gpu_rew(rewards)

            next_obs_t  = _to_gpu_obs(next_obs_np)
            next_done_t = _to_gpu_done(next_done_np)
            global_step += E

        # ----------------------------------------------------------------
        # Compute GAE advantages
        # ----------------------------------------------------------------
        with torch.no_grad():
            obs_flat = next_obs_t.reshape(E * N, obs_dim)
            task_id_batch = torch.full((E,), current_task_id,
                                       dtype=torch.long, device=device)
            next_value = agent.get_value(obs_flat, task_id_batch, n_robots=N)

            # Normalize returns with PopArt
            raw_advantages, raw_returns = compute_gae(
                buf_rewards, buf_values, next_value, buf_dones,
                args.gamma, args.gae_lambda
            )

            # Normalize advantages using PopArt sigma
            norm_returns = popart.normalize(raw_returns.reshape(-1), current_task_id)
            norm_returns = norm_returns.reshape(T, E)
            advantages = raw_advantages  # advantages keep raw scale for PPO ratio

        # ----------------------------------------------------------------
        # PPO update
        # ----------------------------------------------------------------
        # Flatten over (T, E) for minibatch sampling.
        # Policy operates on (T*E*N, obs_dim); critic on (T*E, global_state_dim).
        flat_obs      = buf_obs.reshape(T * E * N, obs_dim)
        flat_actions  = buf_actions.reshape(T * E * N, ROBOT_ACTUATORS)
        flat_logprobs = buf_logprobs.reshape(T * E * N)
        flat_taskids  = buf_taskids.reshape(T * E)          # (T*E,)
        flat_advs     = advantages.reshape(T * E)           # one per env-step
        flat_returns  = norm_returns.reshape(T * E)

        # Expand advantages over N robots (same advantage for all in team)
        flat_advs_per_robot = flat_advs.repeat_interleave(N)  # (T*E*N,)

        clipfracs = []
        for epoch in range(args.num_epochs):
            # Random permutation over env-steps (not robot-steps)
            env_step_indices = torch.randperm(T * E, device=device)

            for start in range(0, T * E, minibatch_size):
                end = start + minibatch_size
                env_idx = env_step_indices[start:end]    # (mb_size,)

                # Expand to robot indices
                robot_idx = (env_idx.unsqueeze(1) * N + torch.arange(N, device=device).unsqueeze(0)).reshape(-1)

                mb_obs      = flat_obs[robot_idx]        # (mb*N, obs_dim)
                mb_actions  = flat_actions[robot_idx]    # (mb*N, 12)
                mb_logprobs = flat_logprobs[robot_idx]   # (mb*N,)
                mb_advs     = flat_advs_per_robot[robot_idx]  # (mb*N,)
                mb_taskids  = flat_taskids[env_idx]      # (mb_size,)
                mb_returns  = flat_returns[env_idx]      # (mb_size,)

                # Forward pass + loss under AMP autocast (fp16 on GPU, no-op on CPU)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    _, new_logprobs, new_entropy, new_values = agent.get_action_and_value(
                        mb_obs, mb_taskids, n_robots=N, action=mb_actions
                    )

                    # PPO clip loss (per robot)
                    log_ratio = new_logprobs - mb_logprobs
                    ratio = log_ratio.exp()

                    # Normalize advantages per minibatch
                    if args.norm_adv:
                        mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                    pg_loss1 = -mb_advs * ratio
                    pg_loss2 = -mb_advs * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (normalized returns)
                    v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                    # Entropy loss (per robot)
                    entropy_loss = new_entropy.mean()

                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                # KL / clipfrac computed in fp32 for numerical accuracy
                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                # Unscale before grad clipping so the clip threshold is in fp32 units
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ----------------------------------------------------------------
        # Update PopArt statistics (after optimizer step)
        # ----------------------------------------------------------------
        with torch.no_grad():
            raw_returns_flat = raw_returns.reshape(T * E)
            popart.update_and_rescale(
                raw_returns_flat,
                current_task_id,
                agent.critic.value_heads[current_task_id],
            )

        # ----------------------------------------------------------------
        # Update curriculum with TD errors
        # ----------------------------------------------------------------
        with torch.no_grad():
            # Re-compute values for TD error tracking (cheap, no grad needed)
            obs_all = flat_obs.reshape(T * E, N, obs_dim)
            obs_all_flat = obs_all.reshape(T * E * N, obs_dim)
            task_id_all = flat_taskids
            vals_for_curriculum = agent.get_value(
                obs_all_flat, task_id_all.repeat_interleave(N)
                if False else task_id_all,  # keep at env-step level
                n_robots=N,
            )
            td_errors = (vals_for_curriculum - raw_returns.reshape(T * E)).cpu().numpy()
        curriculum.update(current_task_id, td_errors)

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        if update % 10 == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            y_pred = buf_values.reshape(-1).cpu().numpy()
            y_true = raw_returns.reshape(-1).cpu().numpy()
            var_y = np.var(y_true)
            explained_var = float(1 - np.var(y_true - y_pred) / (var_y + 1e-8))

            writer.add_scalar("charts/learning_rate",
                               optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/phase", current_phase, global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            # Curriculum probabilities
            task_probs = curriculum.task_probs()
            for t, p in enumerate(task_probs):
                writer.add_scalar(f"curriculum/task_{t}_prob", float(p), global_step)
            writer.add_scalar("curriculum/current_task", current_task_id, global_step)

            # PopArt statistics
            for t in range(args.n_tasks):
                writer.add_scalar(f"popart/mu_task{t}", popart.mu[t].item(), global_step)
                writer.add_scalar(f"popart/sigma_task{t}", popart.sigma[t].item(), global_step)

            # GPU memory
            if device.type == "cuda":
                writer.add_scalar(
                    "gpu/memory_allocated_gb",
                    torch.cuda.memory_allocated() / 1024**3,
                    global_step,
                )
                writer.add_scalar(
                    "gpu/memory_reserved_gb",
                    torch.cuda.memory_reserved() / 1024**3,
                    global_step,
                )

            print(
                f"update={update}/{num_updates}  step={global_step:,}  "
                f"phase={current_phase}  task={TASK_NAMES[current_task_id]}  "
                f"SPS={sps}  pg_loss={pg_loss.item():.4f}  "
                f"v_loss={v_loss.item():.4f}  ent={entropy_loss.item():.4f}"
            )

        # ----------------------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------------------
        if update % args.save_interval == 0:
            ckpt_path = checkpoint_dir / f"ckpt_step{global_step}.pt"
            save_checkpoint(ckpt_path, agent, optimizer, global_step, current_phase, args)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    save_checkpoint(
        checkpoint_dir / "ckpt_final.pt",
        agent, optimizer, global_step, current_phase, args
    )
    print(f"\n[SC-MAPPO] Training complete.  Logs: {log_dir}")

    for env in envs:
        env.close()
    writer.close()


if __name__ == "__main__":
    main()
