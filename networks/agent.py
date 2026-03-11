"""SC-MAPPO Agent: shared policy + centralized critic.

The agent handles all three phases of training:

Phase 1 — Locomotion bootstrap
    Attention is disabled (use_attention=False).
    Only value_heads[0] is used (single head).
    The policy_backbone + policy_head form a standard Gaussian policy.

Phase 2 — Multi-task coordination
    Attention is enabled.
    All four value heads are active.
    policy_backbone lower layers may be frozen by the caller.

Phase 3 — End-to-end fine-tuning
    All layers unfrozen.  Same forward pass as Phase 2.

Forward pass (per robot, batched over B*N):
    1. task_emb = task_embeddings(task_id)           # (B, emb_dim)
    2. proprio_feats = policy_backbone(obs_per_robot) # (B*N, backbone_out)
    3. attn_out = attention(proprio.reshape(B,N,-1), task_emb)  # (B, N, attn_H)
    4. combined = cat(proprio_feats, attn_out.reshape(B*N,-1))  # (B*N, *)
    5. action_mean = policy_head(combined)            # (B*N, 8)
    6. action ~ Normal(action_mean, exp(log_std))
    7. global_state = obs_flat.reshape(B, N*obs_dim)
       value = critic(global_state, task_emb[batch], task_id)  # (B,)

Observation layout fed to this module:
    obs: (B*N, obs_dim_per_robot)  ← full per-robot obs incl. neighbors + task emb
    The 'proprio_dim' slice (first PROPRIO_DIM dims) is the local proprioceptive
    part fed to attention; the full obs is fed to the policy backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from networks.attention import TaskConditionedAttention
from networks.critic import MultiHeadCritic


def _layer_init(layer: nn.Linear, std: float = 1.0, bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


BACKBONE_HIDDEN = 256
ATTN_HIDDEN = 64
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class SCMAPPOAgent(nn.Module):
    """Skill-Conditioned Multi-Agent PPO policy + centralized critic.

    Args:
        obs_dim_per_robot: Full observation dimension per robot (incl. task emb).
        proprio_dim: Proprioceptive-only slice of obs (fed to attention).
        action_dim: Actions per robot (12 for the Go1).
        n_robots: Number of robots in the team.
        n_tasks: Number of coordination tasks.
        task_emb_dim: Task embedding size.
        attn_hidden: Attention hidden dimension.
        critic_hidden: Critic trunk hidden dimension.
    """

    def __init__(
        self,
        obs_dim_per_robot: int,
        proprio_dim: int,
        action_dim: int = 12,
        n_robots: int = 2,
        n_tasks: int = 4,
        task_emb_dim: int = 8,
        attn_hidden: int = ATTN_HIDDEN,
        critic_hidden: int = BACKBONE_HIDDEN,
    ) -> None:
        super().__init__()
        self.n_robots = n_robots
        self.n_tasks = n_tasks
        self.task_emb_dim = task_emb_dim
        self.proprio_dim = proprio_dim
        self.obs_dim = obs_dim_per_robot
        self.action_dim = action_dim

        # ---- Learned per-task embeddings ----
        self.task_embeddings = nn.Embedding(n_tasks, task_emb_dim)
        nn.init.orthogonal_(self.task_embeddings.weight)

        # ---- Policy backbone (may be partially frozen in Phase 2) ----
        self.policy_backbone = nn.Sequential(
            _layer_init(nn.Linear(obs_dim_per_robot, BACKBONE_HIDDEN)),
            nn.Tanh(),
            _layer_init(nn.Linear(BACKBONE_HIDDEN, BACKBONE_HIDDEN)),
            nn.Tanh(),
        )

        # ---- Task-conditioned cross-robot attention ----
        self.attention = TaskConditionedAttention(
            proprio_dim=proprio_dim,
            task_emb_dim=task_emb_dim,
            hidden_dim=attn_hidden,
        )
        self._use_attention: bool = False  # toggled by training loop

        # Policy head: takes backbone output concatenated with attention output
        policy_head_in = BACKBONE_HIDDEN + attn_hidden
        self.policy_head = nn.Sequential(
            _layer_init(nn.Linear(policy_head_in, BACKBONE_HIDDEN)),
            nn.Tanh(),
            _layer_init(nn.Linear(BACKBONE_HIDDEN, action_dim), std=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # ---- Centralized multi-head critic ----
        global_state_dim = n_robots * obs_dim_per_robot
        self.critic = MultiHeadCritic(
            global_state_dim=global_state_dim,
            task_emb_dim=task_emb_dim,
            n_tasks=n_tasks,
            trunk_hidden=critic_hidden,
        )

    # ------------------------------------------------------------------
    # Phase transitions (called by training loop)
    # ------------------------------------------------------------------

    def enable_attention(self) -> None:
        """Activate attention module (Phase 2+)."""
        self._use_attention = True

    def freeze_backbone_lower(self) -> None:
        """Freeze the first layer of the policy backbone (Phase 2 transfer)."""
        first_layer = self.policy_backbone[0]
        for p in first_layer.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (Phase 3 fine-tuning)."""
        for p in self.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def get_action_and_value(
        self,
        obs: torch.Tensor,          # (B*N, obs_dim)
        task_id_tensor: torch.Tensor,  # (B,) long — one task id per batch element
        n_robots: int | None = None,
        deterministic: bool = False,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute value, log-prob, entropy.

        Args:
            obs: Stacked per-robot observations, batch-major.
                 Shape: (B*N, obs_dim_per_robot).
            task_id_tensor: Long tensor of task IDs, one per *batch env*.
                            Shape: (B,).
            n_robots: Override n_robots (uses self.n_robots by default).
            deterministic: If True, use action mean (no sampling).
            action: If provided, evaluate log-prob of this action instead
                    of sampling (used during PPO update).

        Returns:
            action:   (B*N, action_dim)
            log_prob: (B*N,)
            entropy:  (B*N,)
            value:    (B,)  — one value per batch environment (not per robot)
        """
        N = n_robots if n_robots is not None else self.n_robots
        BN = obs.shape[0]
        B = BN // N

        # Task embeddings: (B, task_emb_dim), then repeat for all N robots
        task_emb = self.task_embeddings(task_id_tensor)   # (B, emb_dim)
        task_emb_repeated = task_emb.repeat_interleave(N, dim=0)  # (B*N, emb_dim)

        # Policy backbone
        backbone_out = self.policy_backbone(obs)   # (B*N, BACKBONE_HIDDEN)

        # Attention (disabled in Phase 1 → zero contribution)
        if self._use_attention:
            proprio = obs[:, : self.proprio_dim]                   # (B*N, proprio_dim)
            proprio_3d = proprio.reshape(B, N, self.proprio_dim)   # (B, N, proprio_dim)
            attn_out = self.attention(proprio_3d, task_emb)        # (B, N, attn_H)
            attn_flat = attn_out.reshape(BN, -1)                   # (B*N, attn_H)
        else:
            attn_flat = torch.zeros(
                BN, self.attention.hidden_dim, device=obs.device, dtype=obs.dtype
            )

        combined = torch.cat([backbone_out, attn_flat], dim=-1)  # (B*N, H+attn_H)
        action_mean = self.policy_head(combined)                  # (B*N, action_dim)

        # Gaussian policy
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        dist = Normal(action_mean, log_std.exp())

        if action is None:
            action = action_mean if deterministic else dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)   # (B*N,)
        entropy = dist.entropy().sum(dim=-1)            # (B*N,)

        # Centralized critic: global state = all robot obs flattened per env
        global_state = obs.reshape(B, N * self.obs_dim)  # (B, N*obs_dim)
        # Use task_id from the first element (all envs in a batch share same task)
        task_id_scalar = int(task_id_tensor[0].item())
        value = self.critic(global_state, task_emb, task_id_scalar)  # (B,)

        return action, log_prob, entropy, value

    def get_value(
        self,
        obs: torch.Tensor,
        task_id_tensor: torch.Tensor,
        n_robots: int | None = None,
    ) -> torch.Tensor:
        """Compute value only (no action sampling). Returns (B,)."""
        N = n_robots if n_robots is not None else self.n_robots
        B = obs.shape[0] // N
        task_emb = self.task_embeddings(task_id_tensor)
        global_state = obs.reshape(B, N * self.obs_dim)
        task_id_scalar = int(task_id_tensor[0].item())
        return self.critic(global_state, task_emb, task_id_scalar)
