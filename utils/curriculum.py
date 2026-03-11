"""Critic-uncertainty automatic curriculum.

Task sampling probability is proportional to the variance of recent
TD errors for each task.  Tasks where the critic is most uncertain
(high TD-error variance) are sampled more often, because that is where
the most learning signal remains.

High-level flow in train.py:
    curriculum = CurriculumSampler(n_tasks=4)

    # After each PPO update for task_id:
    td_errors = (values - returns).detach()
    curriculum.update(task_id, td_errors)

    # Before next rollout (Phase 2+ only):
    next_task_id = curriculum.sample()

Phase 1 forces task 0 regardless; the caller controls this.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np


_MIN_PROB = 0.05   # ensure every task gets some probability floor


class CurriculumSampler:
    """Automatic curriculum scheduler based on TD-error variance.

    Args:
        n_tasks: Total number of tasks.
        window: Rolling window size for TD-error variance estimates.
        temperature: Softmax temperature; higher = more uniform distribution.
        min_prob: Minimum probability floor per task (prevents starvation).
    """

    def __init__(
        self,
        n_tasks: int = 4,
        window: int = 100,
        temperature: float = 1.0,
        min_prob: float = _MIN_PROB,
    ) -> None:
        self.n_tasks = n_tasks
        self.temperature = temperature
        self.min_prob = min_prob
        # One deque of recent TD-error variances per task
        self._var_history: list[deque[float]] = [
            deque(maxlen=window) for _ in range(n_tasks)
        ]
        # Seed each task's history with a neutral value so sampling works
        # before any real data is available
        for q in self._var_history:
            q.append(1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, task_id: int, td_errors: np.ndarray) -> None:
        """Record TD-error variance for task_id after a PPO update.

        Args:
            task_id: Which task was just trained.
            td_errors: 1-D array of (value - return) for the task's batch.
        """
        if len(td_errors) > 0:
            variance = float(np.var(td_errors))
            self._var_history[task_id].append(variance)

    def sample(self) -> int:
        """Sample a task index proportional to recent TD-error variance.

        Returns:
            Integer task id in [0, n_tasks).
        """
        variances = np.array(
            [float(np.mean(self._var_history[t])) for t in range(self.n_tasks)],
            dtype=np.float64,
        )
        # Softmax over (variance / temperature)
        logits = variances / (self.temperature + 1e-8)
        logits -= logits.max()           # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()

        # Apply probability floor and renormalise
        probs = np.clip(probs, self.min_prob, None)
        probs /= probs.sum()

        return int(np.random.choice(self.n_tasks, p=probs))

    def task_probs(self) -> np.ndarray:
        """Return current sampling probabilities for all tasks. Shape: (n_tasks,)."""
        variances = np.array(
            [float(np.mean(self._var_history[t])) for t in range(self.n_tasks)],
            dtype=np.float64,
        )
        logits = variances / (self.temperature + 1e-8)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        probs = np.clip(probs, self.min_prob, None)
        probs /= probs.sum()
        return probs.astype(np.float32)
