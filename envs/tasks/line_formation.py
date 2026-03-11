"""Task 1: Line Formation.

All robots walk forward (positive x) while maintaining equal lateral
(y-axis) spacing of DESIRED_SPACING metres between adjacent robots.

Team reward = group_forward_vel + formation_tightness - ctrl_cost
"""
from __future__ import annotations

import numpy as np

from envs.multi_ant_base import MultiAntBase


class LineFormationEnv(MultiAntBase):
    TASK_ID = 1
    DESIRED_SPACING: float = 2.5  # metres between adjacent robots (same as initial)

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("task_id", self.TASK_ID)
        super().__init__(**kwargs)
        self._prev_mean_x: float | None = None

    def _compute_reward(self, actions: np.ndarray) -> tuple[float, dict]:
        positions = [self._robot_pos(i) for i in range(self.n_robots)]
        mean_x = float(np.mean([p[0] for p in positions]))

        # Forward reward: change in mean x position
        if self._prev_mean_x is not None:
            dt = self.model.opt.timestep * self.frame_skip
            fwd_vel = (mean_x - self._prev_mean_x) / dt
        else:
            fwd_vel = 0.0
        self._prev_mean_x = mean_x

        # Formation reward: all robots should be at equal y-spacing
        y_coords = sorted([p[1] for p in positions])
        spacing_errors = [
            abs(y_coords[k + 1] - y_coords[k] - self.DESIRED_SPACING)
            for k in range(len(y_coords) - 1)
        ]
        total_spacing_err = float(sum(spacing_errors))
        formation_reward = float(np.exp(-total_spacing_err))

        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(actions)))

        reward = self.forward_reward_weight * fwd_vel + formation_reward - ctrl_cost
        info = {
            "fwd_vel": fwd_vel,
            "formation_reward": formation_reward,
            "spacing_error": total_spacing_err,
            "ctrl_cost": ctrl_cost,
        }
        return reward, info

    def _reset_task_specific(self) -> None:
        super()._reset_task_specific()
        self._prev_mean_x = None
