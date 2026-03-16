"""Task 3: Cooperative Push.

All robots jointly push a box from its spawn position to a target
location 4 m ahead along the x-axis.

Team reward = box_progress + proximity_bonus - ctrl_cost

box_progress is a *delta* reward: positive only when the box moves
toward the target.  The old formulation used -box_dist (≈ -4 at
episode start), which created large negative returns incompatible
with other tasks' positive baselines and caused PopArt warm-start
mismatch that destroyed locomotion at Phase 2 onset.
"""
from __future__ import annotations

import numpy as np

from envs.multi_ant_base import MultiAntBase

BOX_SPAWN = np.array([2.0, 0.0, 0.15], dtype=np.float32)
BOX_TARGET = np.array([6.0, 0.0, 0.15], dtype=np.float32)
_INITIAL_BOX_DIST = float(np.linalg.norm(BOX_SPAWN[:2] - BOX_TARGET[:2]))  # 4.0 m


class CoopPushEnv(MultiAntBase):
    TASK_ID = 3

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("task_id", self.TASK_ID)
        super().__init__(**kwargs)
        self._prev_box_dist: float = _INITIAL_BOX_DIST

    def _compute_reward(self, actions: np.ndarray) -> tuple[float, dict]:
        box_pos = self._box_pos()
        box_dist = float(np.linalg.norm(box_pos[:2] - BOX_TARGET[:2]))

        # Progress reward: positive only when the box moves toward the target.
        push_reward = self._prev_box_dist - box_dist  # 0 at start, positive on progress
        self._prev_box_dist = box_dist

        # Proximity bonus: encourage robots to stay near the box
        proximity = 0.0
        for i in range(self.n_robots):
            robot_pos = self._robot_pos(i)
            proximity += float(np.exp(-np.linalg.norm(robot_pos - box_pos)))
        proximity /= self.n_robots

        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(actions)))

        reward = push_reward + 0.5 * proximity - ctrl_cost
        info = {
            "box_dist": box_dist,
            "push_progress": push_reward,
            "proximity_bonus": proximity,
            "ctrl_cost": ctrl_cost,
        }
        return reward, info

    def _reset_task_specific(self) -> None:
        """Place the push box at its spawn position."""
        bqs = self._box_qpos_start
        self.data.qpos[bqs : bqs + 3] = BOX_SPAWN
        self.data.qpos[bqs + 3 : bqs + 7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion
        # Zero box velocity
        bvs = self._box_qvel_start
        self.data.qvel[bvs : bvs + 6] = 0.0
        self._prev_box_dist = _INITIAL_BOX_DIST
