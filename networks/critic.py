"""Task-Conditioned Multi-Head Centralized Critic (Innovation 2).

Architecture:
    input  = concat(global_state, task_emb)        # (B, N*obs_dim + task_emb_dim)
    trunk  = Linear → Tanh → Linear → Tanh → ...  # shared representation
    heads  = [Linear(trunk_out, 1) for each task]  # per-task value scalars

During training the head matching the *current* task is selected.
This directly addresses gradient conflicts: task-specific value heads
prevent one task's gradients from corrupting another's value estimates.

The PopArtNormalizer (utils/popart.py) operates on the value heads'
weights *externally* — this module just exposes the heads as a
ModuleList so the normalizer can rescale them in-place.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _layer_init(layer: nn.Linear, std: float = 1.0, bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MultiHeadCritic(nn.Module):
    """Centralized critic with per-task value heads.

    Args:
        global_state_dim: Dimension of the concatenated global state
            (n_robots * obs_dim_per_robot).
        task_emb_dim: Task embedding dimension (default 8).
        n_tasks: Number of tasks / value heads (default 4).
        trunk_hidden: Hidden dimension of the shared trunk MLP (default 256).
    """

    def __init__(
        self,
        global_state_dim: int,
        task_emb_dim: int = 8,
        n_tasks: int = 4,
        trunk_hidden: int = 256,
    ) -> None:
        super().__init__()
        input_dim = global_state_dim + task_emb_dim
        self.n_tasks = n_tasks

        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(input_dim, trunk_hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(trunk_hidden, trunk_hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(trunk_hidden, trunk_hidden)),
            nn.Tanh(),
        )

        # Per-task value heads exposed for PopArt rescaling
        self.value_heads = nn.ModuleList([
            _layer_init(nn.Linear(trunk_hidden, 1), std=1.0)
            for _ in range(n_tasks)
        ])

    def forward(
        self,
        global_state: torch.Tensor,   # (B, N*obs_dim)
        task_emb: torch.Tensor,       # (B, task_emb_dim)
        task_id: int,
    ) -> torch.Tensor:
        """Compute value estimate using the task-specific head.

        Returns:
            Tensor of shape (B,) — scalar value per batch element.
        """
        x = torch.cat([global_state, task_emb], dim=-1)
        features = self.trunk(x)
        value = self.value_heads[task_id](features)   # (B, 1)
        return value.squeeze(-1)                       # (B,)
