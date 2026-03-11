"""Task 0: Follow-the-Leader.

Robot 0 is the leader — it gets rewarded for moving forward (positive x).
Robot 1+ are followers — they get rewarded for staying at a fixed offset
behind the leader (2 m in the -x direction, same y).

Team reward = leader_forward_vel + sum(follower_proximity) - ctrl_cost
"""
from __future__ import annotations

import numpy as np

from envs.multi_ant_base import MultiAntBase


class FollowTheLeaderEnv(MultiAntBase):
    TASK_ID = 0
    # Desired offset from leader to each follower (in world frame)
    FOLLOWER_OFFSET = np.array([-2.0, 0.0, 0.0], dtype=np.float32)

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("task_id", self.TASK_ID)
        super().__init__(**kwargs)
        self._prev_leader_x: float | None = None

    def _compute_reward(self, actions: np.ndarray) -> tuple[float, dict]:
        leader_pos = self._robot_pos(0)

        # Leader forward reward: x-velocity
        if self._prev_leader_x is not None:
            dt = self.model.opt.timestep * self.frame_skip
            leader_fwd = (leader_pos[0] - self._prev_leader_x) / dt
        else:
            leader_fwd = 0.0
        self._prev_leader_x = float(leader_pos[0])

        # Follower proximity reward
        follower_reward = 0.0
        for i in range(1, self.n_robots):
            follower_pos = self._robot_pos(i)
            target = leader_pos + self.FOLLOWER_OFFSET
            dist = float(np.linalg.norm(follower_pos - target))
            follower_reward += float(np.exp(-dist))  # 1 when at target, 0 when far

        follower_reward /= max(self.n_robots - 1, 1)

        # Control cost (penalise large actions)
        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(actions)))

        reward = self.forward_reward_weight * leader_fwd + follower_reward - ctrl_cost
        info = {
            "leader_fwd_vel": leader_fwd,
            "follower_reward": follower_reward,
            "ctrl_cost": ctrl_cost,
        }
        return reward, info

    def _reset_task_specific(self) -> None:
        super()._reset_task_specific()
        self._prev_leader_x = None
