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
import tyro
from gymnasium.vector import AsyncVectorEnv

from envs.make_env import make_env, TASK_NAMES
from envs.multi_ant_base import PROPRIO_DIM, ROBOT_ACTUATORS
from utils.curriculum import CurriculumSampler


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
    num_envs: int = 32
    """Number of parallel environments stepped by AsyncVectorEnv workers.
    Each env runs in its own subprocess (real CPU parallelism).
    Raised from 16 → 32; try 64 if CPU core count allows."""
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
    target_kl: float = 0.01
    """Early stop PPO epochs when approx KL exceeds this value.
    0.01 is a good default: catches phase-transition instability early
    (the primary symptom of Phase 3 optimizer state loss was unchecked KL
    runaway across all 10 epochs before the loss spike became visible).
    Set to 0 to disable."""

    # SC-MAPPO specific
    attn_hidden: int = 64
    critic_hidden: int = 256
    curriculum_temp: float = 1.0
    curriculum_window: int = 100
    """Rolling window length for TD-error variance estimates."""

    # Environment reward shaping
    ctrl_cost_weight: float = 0.01
    """Penalty weight on sum of squared actions per step.  Lower = policy can
    take larger actions early in training (helps with balance bootstrap).
    Default 0.01 (was 0.05 — too high relative to healthy_reward=1.0)."""
    healthy_reward: float = 2.0
    """Per-step bonus for staying alive (z in healthy range).  Higher = stronger
    survival signal relative to ctrl_cost.  Default 2.0 (was 1.0)."""
    frame_skip: int = 5
    """Physics substeps per action.  Lower = more control steps per simulated
    second = much easier to learn balance for a 12-DOF quadruped.
    Default 5 (100 Hz), was 25 (20 Hz) — too coarse to learn from.
    If you raise this back toward 25 for transfer to real hardware, also
    raise phase1_steps proportionally."""
    reset_noise_scale: float = 0.02
    """Uniform noise std added to qpos/qvel at episode reset.
    Default 0.02 (was 0.1 — too large; started robots in destabilised
    joint configurations that caused falls within 6 steps on average)."""

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
    import torch
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

def _warmstart_new_task_heads(
    agent,
    popart,
    n_tasks: int,
) -> None:
    """Copy task-0 value head weights and PopArt stats to all other task heads.

    Without this, untrained heads produce near-zero values against returns of
    ~200+ (learned locomotion), creating value loss on the order of 20,000 that
    blasts the shared trunk and destroys task-0's estimates as well.
    """
    import torch
    with torch.no_grad():
        src = agent.critic.value_heads[0]
        for t in range(1, n_tasks):
            agent.critic.value_heads[t].weight.data.copy_(src.weight.data)
            agent.critic.value_heads[t].bias.data.copy_(src.bias.data)
            popart.mu[t]    = popart.mu[0].clone()
            popart.sigma[t] = popart.sigma[0].clone()
            popart.nu[t]    = popart.nu[0].clone()
    print(
        f"  Warm-started task heads 1-{n_tasks-1} from task 0 "
        f"(mu={popart.mu[0].item():.2f}, sigma={popart.sigma[0].item():.2f})"
    )


def transition_to_phase2(
    agent: SCMAPPOAgent,
    optimizer: optim.Optimizer,
    popart,
    args: Args,
) -> None:
    """Freeze first backbone layer, enable attention, warm-start task heads."""
    print("\n[SC-MAPPO] Advancing to Phase 2: multi-task coordination")
    agent.enable_attention()
    agent.freeze_backbone_lower()
    _warmstart_new_task_heads(agent, popart, args.n_tasks)
    # Rebuild optimizer param groups to exclude the now-frozen backbone[0].
    # Do NOT clear optimizer.state — it maps param tensor objects → Adam
    # step/exp_avg/exp_avg_sq.  Clearing it loses all Phase-1 momentum,
    # which resets v_t → 0 and makes the effective lr on the first Phase-2
    # update ≈ lr / sqrt(eps) ≈ 3× Phase 1 lr — a crash, not a smooth
    # transition.  Keeping state is safe: after clear+add, non-frozen param
    # tensors are in the new group and Adam finds their existing state by
    # tensor identity.  Backbone[0] state entries are pruned below (orphaned
    # keys cause a KeyError in optimizer.state_dict()).
    optimizer.param_groups.clear()
    optimizer.add_param_group({"params": [p for p in agent.parameters() if p.requires_grad]})
    # Prune optimizer.state entries for params no longer in any param group
    # (i.e. backbone[0], which was just frozen).  PyTorch's state_dict()
    # builds a param→index mapping from param_groups and then looks up every
    # key in optimizer.state against it — an orphaned key raises KeyError.
    active_params = {p for pg in optimizer.param_groups for p in pg["params"]}
    for k in list(optimizer.state.keys()):
        if k not in active_params:
            del optimizer.state[k]


def transition_to_phase3(
    agent: SCMAPPOAgent,
    optimizer: optim.Optimizer,
    args: Args,
) -> None:
    """Unfreeze all layers, reduce LR by 10×.

    Deliberately preserves Adam first/second-moment state for all params
    that were active in Phase 2.  Clearing state (the naive approach) resets
    v_t → 0, making the effective step size lr/10 / sqrt(eps) ≈ 3× the Phase 1
    lr on the first update — the opposite of the intended reduction — which
    crashes episodic length and spikes value/policy loss.
    """
    print("\n[SC-MAPPO] Advancing to Phase 3: end-to-end fine-tuning")
    agent.unfreeze_all()
    new_lr = args.learning_rate / 10.0

    # Lower lr for all existing param groups (Adam state is preserved).
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr

    # backbone[0] was frozen in Phase 2 → not in any existing param group.
    # Add it now so it gets trained in Phase 3.  Starting without Adam state
    # is fine here: these params have had no Phase-2 updates, so there is no
    # accumulated momentum to preserve.
    existing_ids = {id(p) for pg in optimizer.param_groups for p in pg["params"]}
    new_params = [p for p in agent.parameters() if id(p) not in existing_ids]
    if new_params:
        optimizer.add_param_group({"params": new_params, "lr": new_lr})
        print(f"  Added {len(new_params)} newly-unfrozen params to optimizer")

    print(f"  Learning rate reduced to {new_lr:.2e}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    agent: SCMAPPOAgent,
    optimizer: optim.Optimizer,
    popart,
    global_step: int,
    current_phase: int,
    args: Args,
) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "popart": popart.state_dict(),
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
    popart,
    device: torch.device,
    target_phase: int,
) -> tuple[int, int]:
    """Load checkpoint. Returns (global_step, current_phase).

    Skips optimizer state when the target phase differs from the checkpoint
    phase — the param group layout changes at phase boundaries, so loading
    a Phase-1 optimizer state into a Phase-2 optimizer would either crash or
    silently assign stale Adam momentum to the wrong parameters.
    """
    import torch
    ckpt = torch.load(path, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent"])

    loaded_phase = ckpt["current_phase"]
    if loaded_phase == target_phase:
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        # Phase boundary: param groups changed (e.g. Phase 2 has backbone[0]
        # frozen, Phase 3 doesn't).  Attempt to load state anyway; if the
        # param counts differ PyTorch will raise a ValueError and we fall back
        # to a cold optimizer.  The cold-start issue is mitigated at run-time
        # by transition_to_phase3 preserving state for the auto-advance path.
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print(
                f"  Loaded optimizer state across phase boundary "
                f"(ckpt phase={loaded_phase} → target phase={target_phase})."
            )
        except (ValueError, RuntimeError):
            print(
                f"  Optimizer state shape mismatch across phase boundary "
                f"(ckpt phase={loaded_phase} → target phase={target_phase}) "
                f"— optimizer re-initialized.  Expect a brief loss spike."
            )

    if "popart" in ckpt:
        popart.load_state_dict(ckpt["popart"])
    else:
        print("  Warning: checkpoint has no popart state (old format) — stats reset.")

    print(f"[SC-MAPPO] Loaded checkpoint from {path}")
    print(f"  global_step={ckpt['global_step']}, phase={loaded_phase}")
    return ckpt["global_step"], loaded_phase


# ---------------------------------------------------------------------------
# Vectorised environment helpers
# ---------------------------------------------------------------------------

def _make_vec_env(task_id: int, args: Args, seed_offset: int = 0) -> AsyncVectorEnv:
    """Spin up num_envs parallel workers for the given task."""
    return AsyncVectorEnv([
        make_env(
            task_id=task_id,
            n_robots=args.n_robots,
            seed=args.seed + i + seed_offset,
            task_embedding_dim=args.task_emb_dim,
            include_cfrc_ext=args.include_cfrc_ext,
            ctrl_cost_weight=args.ctrl_cost_weight,
            healthy_reward=args.healthy_reward,
            frame_skip=args.frame_skip,
            reset_noise_scale=args.reset_noise_scale,
        )
        for i in range(args.num_envs)
    ])


def _log_vec_episodes(
    infos: dict,
    writer: SummaryWriter,
    task_id: int,
    global_step: int,
) -> None:
    """Log episodic return / length from gymnasium vectorised env infos.

    gymnasium >= 1.0 stores per-env episode stats as:
        infos["episode"]["r"]  ->  (num_envs,) float array
        infos["_episode"]      ->  (num_envs,) bool mask (True = episode ended)
    """
    ep = infos.get("episode")
    if ep is None:
        return
    mask = infos.get("_episode", np.ones(args_n_envs, dtype=bool))
    rs = ep.get("r", [])
    ls = ep.get("l", [])
    for valid, r, l in zip(mask, rs, ls):
        if valid and r is not None:
            writer.add_scalar(
                f"charts/episodic_return_task{task_id}", float(r), global_step
            )
            writer.add_scalar(
                f"charts/episodic_length_task{task_id}", float(l), global_step
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Module-level sentinel used by _log_vec_episodes (set in main before rollout)
args_n_envs: int = 16


def main() -> None:  # noqa: C901
    # Deferred torch imports: worker subprocesses spawned by AsyncVectorEnv
    # re-import this module as __mp_main__ on Windows.  Importing torch at
    # module level causes every worker to load ~2 GB of CUDA DLLs, exhausting
    # the paging file with 32+ workers.  Importing inside main() avoids this.
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    from networks.agent import SCMAPPOAgent
    from utils.popart import PopArtNormalizer

    global args_n_envs
    args = tyro.cli(Args)
    args_n_envs = args.num_envs

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    # Enable cuDNN auto-tuner when not in strict deterministic mode.
    if device.type == "cuda" and not args.torch_deterministic:
        torch.backends.cudnn.benchmark = True

    # AMP: fp16 forward + fp32 param updates (CUDA only).
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
    # Build environments — AsyncVectorEnv runs each env in its own
    # subprocess so all num_envs MuJoCo instances step in parallel.
    # Phase 1: all workers run task 0 (follow-the-leader).
    # ------------------------------------------------------------------
    print(f"[SC-MAPPO] Launching {args.num_envs} parallel env workers (task 0)…")
    vec_env = _make_vec_env(task_id=0, args=args)
    obs_dict, _ = vec_env.reset()

    obs_dim   = obs_dict["obs"].shape[-1]        # obs_dim_per_robot
    # Derive proprio_dim from a single env (call returns list; take first)
    proprio_dim = vec_env.call("proprio_dim")[0]  # int attribute

    # Current task being run by vec_env (tracked to detect task switches)
    vec_env_task_id: int = 0

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

    # Apply phase-specific setup BEFORE building the optimizer so the
    # optimizer's param groups match what was saved in the checkpoint.
    # (transition_to_phase2 freezes some params and saves a reduced optimizer;
    # loading that checkpoint into a full-param optimizer causes a size mismatch.)
    current_phase = args.phase
    if current_phase >= 2:
        agent.enable_attention()
        agent.freeze_backbone_lower()
    if current_phase == 3:
        agent.unfreeze_all()

    # Build optimizer AFTER phase setup so param groups match the checkpoint.
    trainable = [p for p in agent.parameters() if p.requires_grad]
    lr = args.learning_rate if current_phase < 3 else args.learning_rate / 10.0
    optimizer = optim.Adam(trainable, lr=lr, eps=1e-5)

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

    if args.checkpoint:
        global_step, loaded_phase = load_checkpoint(
            args.checkpoint, agent, optimizer, popart, device, target_phase=args.phase
        )
        if loaded_phase != args.phase:
            print(f"  Transitioning phase {loaded_phase} → {args.phase}")
            # Warm-start new task heads when jumping from Phase 1 → 2
            if loaded_phase < 2 <= args.phase:
                _warmstart_new_task_heads(agent, popart, args.n_tasks)

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

    # Pinned-memory staging for fast CPU→GPU obs transfer.
    # Page-locked memory lets the DMA engine copy asynchronously while the
    # CPU dispatches env worker results.
    if device.type == "cuda":
        _obs_pin = torch.zeros((E, N, obs_dim), dtype=torch.float32).pin_memory()
    else:
        _obs_pin = None

    def _obs_to_device(arr: np.ndarray) -> torch.Tensor:
        if _obs_pin is not None:
            _obs_pin.copy_(torch.from_numpy(np.ascontiguousarray(arr)))
            return _obs_pin.to(device, non_blocking=True)
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    # Seed initial obs from the already-reset vec_env
    next_obs_np  = obs_dict["obs"]                         # (E, N, obs_dim)
    next_done_np = np.zeros(E, dtype=np.float32)

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
        f"batch_size={batch_size:,}  (num_envs={E}, num_steps={T})"
    )

    checkpoint_dir = log_dir / "checkpoints"
    start_time = time.time()
    rollout_time_ema = 0.0   # exponential moving average for display
    update_time_ema  = 0.0

    for update in range(1, num_updates + 1):

        # ----------------------------------------------------------------
        # Auto-advance phases
        # ----------------------------------------------------------------
        if current_phase == 1 and global_step >= args.phase1_steps:
            current_phase = 2
            transition_to_phase2(agent, optimizer, popart, args)
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

        # ----------------------------------------------------------------
        # Task switch — close old vec_env, open new one for the new task.
        # Happens at most once per rollout (between rollouts), so the cost
        # (~0.1–0.2 s for 32 envs) is amortised over num_steps env steps.
        # ----------------------------------------------------------------
        if current_task_id != vec_env_task_id:
            vec_env.close()
            vec_env = _make_vec_env(
                current_task_id, args, seed_offset=update * 1000
            )
            obs_dict, _ = vec_env.reset()
            next_obs_np  = obs_dict["obs"]
            next_done_np = np.zeros(E, dtype=np.float32)
            vec_env_task_id = current_task_id

        # ----------------------------------------------------------------
        # Inject current task embedding into all subprocess envs so it
        # appears in their observations.  One IPC round-trip per rollout.
        # ----------------------------------------------------------------
        with torch.no_grad():
            task_id_t = torch.tensor([current_task_id], device=device, dtype=torch.long)
            task_emb_np = agent.task_embeddings(task_id_t).squeeze(0).cpu().numpy()
        vec_env.call("set_task_emb", task_emb_np)  # broadcasts to all workers

        # ----------------------------------------------------------------
        # Collect rollout
        # Rollout pattern: GPU inference → send actions to workers →
        # workers step in parallel → gather results → repeat.
        # ----------------------------------------------------------------
        rollout_t0 = time.perf_counter()

        for step in range(T):
            buf_dones[step] = torch.from_numpy(next_done_np).to(device, non_blocking=True)
            buf_taskids[step] = current_task_id

            obs_t = _obs_to_device(next_obs_np)   # (E, N, obs_dim) on device
            buf_obs[step] = obs_t

            with torch.no_grad():
                obs_flat = obs_t.reshape(E * N, obs_dim)
                task_id_batch = torch.full((E,), current_task_id,
                                           dtype=torch.long, device=device)
                actions, log_probs, _, values = agent.get_action_and_value(
                    obs_flat, task_id_batch, n_robots=N
                )

            buf_actions[step]  = actions.reshape(E, N, ROBOT_ACTUATORS)
            buf_logprobs[step] = log_probs.reshape(E, N)
            buf_values[step]   = values   # (E,)

            # Vectorised step — all E envs step in parallel across workers.
            actions_np = actions.reshape(E, N, ROBOT_ACTUATORS).cpu().numpy()
            obs_dict, rewards_np, terms_np, truncs_np, infos = vec_env.step(actions_np)

            # AsyncVectorEnv auto-resets terminated envs; obs_dict already
            # contains the fresh reset obs for done envs.
            next_obs_np  = obs_dict["obs"]                               # (E, N, obs_dim)
            next_done_np = (terms_np | truncs_np).astype(np.float32)    # (E,)

            buf_rewards[step] = torch.from_numpy(rewards_np).to(device, non_blocking=True)

            # Episode stats (gymnasium >= 1.0 puts them in infos["episode"])
            _log_vec_episodes(infos, writer, current_task_id, global_step)

            global_step += E

        rollout_time = time.perf_counter() - rollout_t0
        rollout_time_ema = 0.9 * rollout_time_ema + 0.1 * rollout_time

        # ----------------------------------------------------------------
        # Compute GAE advantages
        # ----------------------------------------------------------------
        with torch.no_grad():
            obs_flat = _obs_to_device(next_obs_np).reshape(E * N, obs_dim)
            task_id_batch = torch.full((E,), current_task_id,
                                       dtype=torch.long, device=device)
            next_value = agent.get_value(obs_flat, task_id_batch, n_robots=N)

            raw_advantages, raw_returns = compute_gae(
                buf_rewards, buf_values, next_value, buf_dones,
                args.gamma, args.gae_lambda
            )

            norm_returns = popart.normalize(raw_returns.reshape(-1), current_task_id)
            norm_returns = norm_returns.reshape(T, E)
            advantages = raw_advantages

        # ----------------------------------------------------------------
        # PPO update
        # ----------------------------------------------------------------
        update_t0 = time.perf_counter()

        flat_obs      = buf_obs.reshape(T * E * N, obs_dim)
        flat_actions  = buf_actions.reshape(T * E * N, ROBOT_ACTUATORS)
        flat_logprobs = buf_logprobs.reshape(T * E * N)
        flat_taskids  = buf_taskids.reshape(T * E)
        flat_advs     = advantages.reshape(T * E)
        flat_returns  = norm_returns.reshape(T * E)

        # Expand advantages over N robots (same advantage for whole team)
        flat_advs_per_robot = flat_advs.repeat_interleave(N)  # (T*E*N,)

        clipfracs = []
        for epoch in range(args.num_epochs):
            env_step_indices = torch.randperm(T * E, device=device)

            for start in range(0, T * E, minibatch_size):
                end = start + minibatch_size
                env_idx = env_step_indices[start:end]

                robot_idx = (
                    env_idx.unsqueeze(1) * N
                    + torch.arange(N, device=device).unsqueeze(0)
                ).reshape(-1)

                mb_obs      = flat_obs[robot_idx]
                mb_actions  = flat_actions[robot_idx]
                mb_logprobs = flat_logprobs[robot_idx]
                mb_advs     = flat_advs_per_robot[robot_idx]
                mb_taskids  = flat_taskids[env_idx]
                mb_returns  = flat_returns[env_idx]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    _, new_logprobs, new_entropy, new_values = agent.get_action_and_value(
                        mb_obs, mb_taskids, n_robots=N, action=mb_actions
                    )

                    log_ratio = new_logprobs - mb_logprobs
                    ratio = log_ratio.exp()

                    if args.norm_adv:
                        mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                    pg_loss1 = -mb_advs * ratio
                    pg_loss2 = -mb_advs * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                    entropy_loss = new_entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            if args.target_kl > 0 and approx_kl > args.target_kl:
                break

        update_time = time.perf_counter() - update_t0
        update_time_ema = 0.9 * update_time_ema + 0.1 * update_time

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
            obs_all_flat = buf_obs.reshape(T * E * N, obs_dim)
            task_id_all  = buf_taskids.reshape(T * E)
            vals_for_curriculum = agent.get_value(
                obs_all_flat, task_id_all,
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

            # Timing breakdown — tells you whether env or GPU is the bottleneck
            total_t = rollout_time_ema + update_time_ema
            writer.add_scalar("perf/rollout_time_s", rollout_time_ema, global_step)
            writer.add_scalar("perf/update_time_s",  update_time_ema,  global_step)
            writer.add_scalar("perf/rollout_pct",
                               100.0 * rollout_time_ema / (total_t + 1e-9), global_step)

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

            rollout_pct = 100.0 * rollout_time_ema / (total_t + 1e-9)
            print(
                f"update={update}/{num_updates}  step={global_step:,}  "
                f"phase={current_phase}  task={TASK_NAMES[current_task_id]}  "
                f"SPS={sps}  pg={pg_loss.item():.4f}  "
                f"v={v_loss.item():.4f}  ent={entropy_loss.item():.4f}  "
                f"rollout={rollout_time_ema:.2f}s ({rollout_pct:.0f}%)  "
                f"update={update_time_ema:.2f}s"
            )

        # ----------------------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------------------
        if update % args.save_interval == 0:
            ckpt_path = checkpoint_dir / f"ckpt_step{global_step}.pt"
            save_checkpoint(ckpt_path, agent, optimizer, popart, global_step, current_phase, args)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    save_checkpoint(
        checkpoint_dir / "ckpt_final.pt",
        agent, optimizer, popart, global_step, current_phase, args
    )
    print(f"\n[SC-MAPPO] Training complete.  Logs: {log_dir}")

    vec_env.close()
    writer.close()


if __name__ == "__main__":
    main()
