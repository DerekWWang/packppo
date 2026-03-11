"""Environment factory following CleanRL's make_env() pattern.

Usage:
    thunk = make_env(task_id=0, n_robots=2, seed=42)
    env = thunk()
    obs, info = env.reset()
"""
from __future__ import annotations

from typing import Callable

import gymnasium as gym

from envs.tasks.follow_the_leader import FollowTheLeaderEnv
from envs.tasks.line_formation import LineFormationEnv
from envs.tasks.v_formation import VFormationEnv
from envs.tasks.coop_push import CoopPushEnv
from envs.multi_ant_base import MultiAntBase

TASK_ENV_MAP: dict[int, type[MultiAntBase]] = {
    0: FollowTheLeaderEnv,
    1: LineFormationEnv,
    2: VFormationEnv,
    3: CoopPushEnv,
}

TASK_NAMES: dict[int, str] = {
    0: "follow_leader",
    1: "line_formation",
    2: "v_formation",
    3: "coop_push",
}


def make_env(
    task_id: int,
    n_robots: int = 2,
    seed: int = 0,
    task_embedding_dim: int = 8,
    render_mode: str | None = None,
    include_cfrc_ext: bool = False,
) -> Callable[[], MultiAntBase]:
    """Return a thunk (zero-arg callable) that creates and seeds an env.

    Compatible with CleanRL's manual env-list pattern.

    Args:
        task_id: Integer task identifier (0–3).
        n_robots: Number of robots (2 or 3).
        seed: Random seed for the environment.
        task_embedding_dim: Dimension of the task embedding vector.
        render_mode: None, "human", or "rgb_array".
        include_cfrc_ext: Include raw contact forces in obs (78 dims, unscaled).
            Disabled by default — foot contact booleans are sufficient for
            locomotion and cfrc_ext values can be orders of magnitude larger
            than other obs features, hurting early training.

    Returns:
        A callable that returns a ready-to-use MultiAntBase instance.
    """
    if task_id not in TASK_ENV_MAP:
        raise ValueError(
            f"Unknown task_id={task_id}. Choose from {list(TASK_ENV_MAP.keys())}."
        )

    def thunk() -> MultiAntBase:
        env_cls = TASK_ENV_MAP[task_id]
        env = env_cls(
            n_robots=n_robots,
            task_embedding_dim=task_embedding_dim,
            render_mode=render_mode,
            include_cfrc_ext=include_cfrc_ext,
        )
        env.reset(seed=seed)
        return env

    return thunk
