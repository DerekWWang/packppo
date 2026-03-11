"""PopArt: Preserving Outputs Precisely while Adaptively Rescaling Targets.

Reference: van Hasselt et al., "Learning values across many orders of
magnitude" (NeurIPS 2016).

Key idea: maintain per-task running mean (mu) and std (sigma) so that
value targets are always normalized before computing the MSE loss.
When statistics change, the corresponding value head weights are rescaled
so the *output* of the head stays the same (hence "Preserving Outputs
Precisely").

Usage in train.py:
    normalizer = PopArtNormalizer(n_tasks=4)

    # Before value loss:
    norm_returns = normalizer.normalize(raw_returns, task_id)

    # After PPO value update, update statistics and rescale head weights:
    normalizer.update_and_rescale(raw_returns, task_id, value_head_linear)
"""
from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn


class PopArtNormalizer(nn.Module):
    """Per-task running mean/std normalizer with output-preserving weight updates."""

    def __init__(
        self,
        n_tasks: int,
        beta: float = 1e-3,          # EMA decay for statistics
        epsilon: float = 1e-5,       # numerical stability
        min_std: float = 1e-2,       # clip sigma from below
    ) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.beta = beta
        self.epsilon = epsilon
        self.min_std = min_std

        # Running statistics (not learnable parameters)
        self.register_buffer("mu", torch.zeros(n_tasks))
        self.register_buffer("sigma", torch.ones(n_tasks))
        self.register_buffer("nu", torch.ones(n_tasks))   # second moment

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, returns: torch.Tensor, task_id: int) -> torch.Tensor:
        """Normalize raw returns using running stats for task_id.

        Args:
            returns: (B,) raw undiscounted or discounted returns.
            task_id: which task's statistics to use.

        Returns:
            (B,) normalized returns.
        """
        return (returns - self.mu[task_id]) / (self.sigma[task_id] + self.epsilon)

    def denormalize(self, values: torch.Tensor, task_id: int) -> torch.Tensor:
        """Convert normalized value estimates back to reward scale."""
        return values * (self.sigma[task_id] + self.epsilon) + self.mu[task_id]

    @torch.no_grad()
    def update_and_rescale(
        self,
        returns: torch.Tensor,       # (B,) raw returns for this task
        task_id: int,
        value_head: nn.Linear,       # the Linear layer for this task's value head
    ) -> None:
        """Update running statistics and rescale value_head weights.

        Must be called *after* the optimizer step so we don't corrupt
        the gradients used in this update.

        Args:
            returns: Raw returns from the current rollout batch.
            task_id: Task whose statistics should be updated.
            value_head: The task-specific nn.Linear(trunk_dim, 1).
        """
        old_sigma = self.sigma[task_id].clone()
        old_mu = self.mu[task_id].clone()

        # EMA update of first and second moments
        batch_mean = returns.mean()
        batch_sq_mean = (returns ** 2).mean()

        self.mu[task_id] = (1 - self.beta) * self.mu[task_id] + self.beta * batch_mean
        self.nu[task_id] = (1 - self.beta) * self.nu[task_id] + self.beta * batch_sq_mean
        new_sigma = torch.sqrt(
            torch.clamp(self.nu[task_id] - self.mu[task_id] ** 2, min=self.min_std ** 2)
        )
        self.sigma[task_id] = new_sigma

        # Rescale value_head weights so output is preserved:
        #   new_W = old_W * (old_sigma / new_sigma)
        #   new_b = (old_b * old_sigma + old_mu - new_mu) / new_sigma
        scale = old_sigma / (new_sigma + self.epsilon)
        value_head.weight.data.mul_(scale)
        if value_head.bias is not None:
            value_head.bias.data.mul_(scale).add_(
                (old_mu - self.mu[task_id]) / (new_sigma + self.epsilon)
            )
