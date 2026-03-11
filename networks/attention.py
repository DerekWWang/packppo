"""Task-Conditioned Cross-Robot Attention (Innovation 1).

Each robot attends to its neighbors.  The query and key projections are
*additively modulated* by the task embedding, so the same attention
module routes information differently depending on the current task —
without maintaining separate networks per task.

Architecture:
    Q = W_q(proprio) + task_mod_q(task_emb)  # (B, N, H)
    K = W_k(proprio) + task_mod_k(task_emb)  # (B, N, H)
    V = W_v(proprio)                          # (B, N, H)
    A = softmax(Q @ K^T / sqrt(H))           # (B, N, N), diagonal masked
    out = out_proj(A @ V)                     # (B, N, H)

The diagonal is masked so each robot only attends to *neighbors*, not
itself (self-attention is already captured by the policy backbone).

For N=1 (single robot, Phase 1 only) attention output is all zeros.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskConditionedAttention(nn.Module):
    """Single-head cross-robot attention with task-conditioned Q/K modulation.

    Args:
        proprio_dim: Dimension of the per-robot proprioceptive features
            *before* concatenating neighbor info and task embedding.
        task_emb_dim: Dimension of task embedding (default 8).
        hidden_dim: Internal attention dimension H (default 64).
    """

    def __init__(
        self,
        proprio_dim: int,
        task_emb_dim: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5

        # Base Q, K, V projections (shared across tasks)
        self.W_q = nn.Linear(proprio_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(proprio_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(proprio_dim, hidden_dim, bias=False)

        # Task-conditioned additive offsets on Q and K outputs
        # Maps task_emb_dim -> hidden_dim; broadcast over N robots
        self.task_mod_q = nn.Linear(task_emb_dim, hidden_dim)
        self.task_mod_k = nn.Linear(task_emb_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [self.W_q, self.W_k, self.W_v, self.out_proj]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
        # Initialise task modulations near zero so the network starts
        # approximately task-agnostic
        nn.init.zeros_(self.task_mod_q.weight)
        nn.init.zeros_(self.task_mod_q.bias)
        nn.init.zeros_(self.task_mod_k.weight)
        nn.init.zeros_(self.task_mod_k.bias)

    def forward(
        self,
        proprio: torch.Tensor,   # (B, N, proprio_dim)
        task_emb: torch.Tensor,  # (B, task_emb_dim)
    ) -> torch.Tensor:
        """Compute task-conditioned neighbor attention.

        Returns:
            Tensor of shape (B, N, hidden_dim) — attention-weighted
            summary of neighbor features for each robot.
        """
        B, N, _ = proprio.shape

        # Base projections
        Q = self.W_q(proprio)   # (B, N, H)
        K = self.W_k(proprio)   # (B, N, H)
        V = self.W_v(proprio)   # (B, N, H)

        # Task-specific additive offsets (broadcast over N)
        q_mod = self.task_mod_q(task_emb).unsqueeze(1)  # (B, 1, H)
        k_mod = self.task_mod_k(task_emb).unsqueeze(1)  # (B, 1, H)
        Q = Q + q_mod
        K = K + k_mod

        # Scaled dot-product attention scores
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, N, N)

        # Mask self-attention: diagonal = -inf
        diag_mask = torch.eye(N, dtype=torch.bool, device=proprio.device)
        attn_logits = attn_logits.masked_fill(diag_mask.unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)   # (B, N, N)

        # Handle N=1 edge case (softmax of -inf → nan)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_out = torch.bmm(attn_weights, V)    # (B, N, H)
        attn_out = self.out_proj(attn_out)        # (B, N, H)
        return attn_out
