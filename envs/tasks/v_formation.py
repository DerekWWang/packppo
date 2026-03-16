"""Task 2: V-Formation Navigation.

Robot 0 is the apex (leader), flanking robots maintain a V-shape while
the group navigates toward a goal 10 m ahead.

V-shape offsets from leader (in world frame, approximate):
  Robot 1: (-FLANK_D, +FLANK_D, 0)  — left rear flank
  Robot 2: (-FLANK_D, -FLANK_D, 0)  — right rear flank
  (For N=2 only robot 1 flanks)

Team reward = goal_approach + formation_tightness - ctrl_cost
"""
from __future__ import annotations

import numpy as np

from envs.multi_ant_base import MultiAntBase

GOAL = np.array([10.0, 0.0, 0.75], dtype=np.float32)
FLANK_D: float = 1.5  # metres

# Pre-computed offsets for up to 2 flanking robots
_FLANK_OFFSETS = [
    np.array([-FLANK_D, +FLANK_D, 0.0], dtype=np.float32),
    np.array([-FLANK_D, -FLANK_D, 0.0], dtype=np.float32),
]


class VFormationEnv(MultiAntBase):
    TASK_ID = 2

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("task_id", self.TASK_ID)
        super().__init__(**kwargs)
        self._prev_goal_dist: float = float(np.linalg.norm(GOAL[:2]))

    def _compute_reward(self, actions: np.ndarray) -> tuple[float, dict]:
        positions = [self._robot_pos(i) for i in range(self.n_robots)]
        leader_pos = positions[0]

        # Goal approach: progress toward goal (delta reward).
        # Replaces -0.1*goal_dist (≈ -1/step at start with goal 10 m away),
        # which undercut the healthy_reward baseline and caused return-scale
        # mismatch with task 0 at Phase 2 onset.
        goal_dist = float(np.linalg.norm(leader_pos[:2] - GOAL[:2]))
        goal_reward = 0.1 * (self._prev_goal_dist - goal_dist)  # 0 at start, positive on approach
        self._prev_goal_dist = goal_dist

        # Formation reward: flanking robots should be at prescribed offsets
        formation_err = 0.0
        for flanker_idx, offset in zip(range(1, self.n_robots), _FLANK_OFFSETS):
            target = leader_pos + offset
            formation_err += float(np.linalg.norm(positions[flanker_idx] - target))

        formation_reward = float(np.exp(-formation_err))

        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(actions)))

        reward = goal_reward + formation_reward - ctrl_cost
        info = {
            "goal_dist": goal_dist,
            "goal_progress": goal_reward,
            "formation_err": formation_err,
            "formation_reward": formation_reward,
            "ctrl_cost": ctrl_cost,
        }
        return reward, info

    def _reset_task_specific(self) -> None:
        super()._reset_task_specific()
        # Reset prev_goal_dist so first step gives zero progress reward.
        # Must be done after mj_forward (called by parent reset) so xpos is fresh.
        self._prev_goal_dist = float(np.linalg.norm(GOAL[:2]))
