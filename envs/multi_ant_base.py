"""Base multi-robot MuJoCo environment (Unitree Go1 quadruped).

Uses the direct mujoco Python API (not gymnasium.MujocoEnv) so we can
control multiple Go1 robots in a single simulation without fighting the
single-agent assumptions baked into gymnasium's wrappers.

Observation per robot (float32):
  - full qpos (x,y,z,quat,12 joints)     = 19
  - full qvel (6 free + 12 hinge)         = 18
  - 4 foot contacts (binarized)           =  4
  - cfrc_ext (13 bodies × 6)             = 78
  - 3 * (n_robots - 1) neighbor rel-pos  =  3*(N-1)
  - 8-dim task embedding                 =  8
  Total (N=2): 19+18+4+78+3+8 = 130
  Total (N=3): 19+18+4+78+6+8 = 133

Action per robot: 12 continuous torques in [-1, 1]  (3 per leg × 4 legs).
"""
from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ASSETS_DIR = Path(__file__).parent.parent / "assets"

# Per-robot body names for cfrc_ext extraction (Unitree Go1)
ROBOT_BODY_NAMES = [
    "trunk",
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]
CFRC_EXT_BODIES = len(ROBOT_BODY_NAMES)  # 13
CFRC_EXT_DIM = CFRC_EXT_BODIES * 6       # 78

# Base proprioceptive dims (without cfrc_ext, neighbors, or task embedding)
#   full qpos (x,y,z, quat, 12 joints)  = 19
#   full qvel (6 free + 12 hinge)        = 18
#   foot contacts                        = 4
_BASE_PROPRIO_DIM = 41

# Default proprioceptive dim (with cfrc_ext included)
PROPRIO_DIM = _BASE_PROPRIO_DIM + CFRC_EXT_DIM  # 119

# Per-robot DoF counts in qpos/qvel (Go1: 3 joints × 4 legs = 12)
ROBOT_QPOS_LEN = 19   # 7 freejoint + 12 hinge
ROBOT_QVEL_LEN = 18   # 6 free + 12 hinge
ROBOT_ACTUATORS = 12   # one per joint (3 per leg × 4 legs)
SENSORS_PER_ROBOT = 4  # foot touch sensors

# Box (push_box body): 7 qpos (free joint), 6 qvel
BOX_QPOS_LEN = 7
BOX_QVEL_LEN = 6


class MultiAntBase(gym.Env):
    """Abstract base for all multi-ant coordination environments."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        n_robots: int = 2,
        task_id: int = 0,
        task_embedding_dim: int = 8,
        frame_skip: int = 25,
        render_mode: str | None = None,
        healthy_z_range: tuple[float, float] = (0.195, 0.75),
        reset_noise_scale: float = 0.1,
        ctrl_cost_weight: float = 0.05,
        forward_reward_weight: float = 1.0,
        healthy_reward: float = 1.0,
        max_episode_steps: int = 1000,
        include_cfrc_ext: bool = True,
    ) -> None:
        super().__init__()

        self.n_robots = n_robots
        self.task_id = task_id
        self.task_embedding_dim = task_embedding_dim
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.healthy_z_range = healthy_z_range
        self.reset_noise_scale = reset_noise_scale
        self.ctrl_cost_weight = ctrl_cost_weight
        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward_value = healthy_reward
        self.max_episode_steps = max_episode_steps
        self.include_cfrc_ext = include_cfrc_ext

        # Proprioceptive dim depends on cfrc_ext setting
        self.proprio_dim = _BASE_PROPRIO_DIM + (CFRC_EXT_DIM if include_cfrc_ext else 0)

        # External handle: training loop writes the current task embedding here
        # so it gets included in observations.  Shape: (task_embedding_dim,)
        self._task_emb: np.ndarray = np.zeros(task_embedding_dim, dtype=np.float32)

        xml_path = ASSETS_DIR / f"multi_go1_{n_robots}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(
                f"MuJoCo XML not found: {xml_path}. "
                f"Supported n_robots values: 2, 3."
            )
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # Cache body IDs, qpos/qvel slice starts, actuator starts
        self._robot_body_ids: list[int] = []
        self._robot_qpos_starts: list[int] = []
        self._robot_qvel_starts: list[int] = []
        self._robot_act_starts: list[int] = []
        self._robot_cfrc_body_ids: list[list[int]] = []
        for i in range(n_robots):
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"trunk_r{i}"
            )
            if bid < 0:
                raise RuntimeError(f"Body 'trunk_r{i}' not found in XML.")
            self._robot_body_ids.append(bid)
            self._robot_qpos_starts.append(i * ROBOT_QPOS_LEN)
            self._robot_qvel_starts.append(i * ROBOT_QVEL_LEN)
            self._robot_act_starts.append(i * ROBOT_ACTUATORS)

            # Cache all body IDs for cfrc_ext extraction
            cfrc_ids: list[int] = []
            for bname in ROBOT_BODY_NAMES:
                cfrc_bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"{bname}_r{i}"
                )
                if cfrc_bid < 0:
                    raise RuntimeError(f"Body '{bname}_r{i}' not found in XML.")
                cfrc_ids.append(cfrc_bid)
            self._robot_cfrc_body_ids.append(cfrc_ids)

        self._box_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "push_box"
        )
        self._box_qpos_start = n_robots * ROBOT_QPOS_LEN
        self._box_qvel_start = n_robots * ROBOT_QVEL_LEN

        # Observation and action spaces
        self._neighbor_dim = 3 * (n_robots - 1)
        self.obs_dim_per_robot = self.proprio_dim + self._neighbor_dim + task_embedding_dim
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(
                -np.inf,
                np.inf,
                shape=(n_robots, self.obs_dim_per_robot),
                dtype=np.float32,
            ),
            "task_id": spaces.Discrete(4),
        })
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(n_robots, ROBOT_ACTUATORS), dtype=np.float32
        )

        # MuJoCo renderer (lazy init)
        self._renderer: mujoco.Renderer | None = None

        # Track episode statistics
        self._episode_return: float = 0.0
        self._episode_length: int = 0

    # ------------------------------------------------------------------
    # Low-level accessors
    # ------------------------------------------------------------------

    def _robot_qpos(self, i: int) -> np.ndarray:
        s = self._robot_qpos_starts[i]
        return self.data.qpos[s : s + ROBOT_QPOS_LEN]

    def _robot_qvel(self, i: int) -> np.ndarray:
        s = self._robot_qvel_starts[i]
        return self.data.qvel[s : s + ROBOT_QVEL_LEN]

    def _robot_foot_contacts(self, i: int) -> np.ndarray:
        """Binarized touch sensor readings for robot i (4 sensors)."""
        s = i * SENSORS_PER_ROBOT
        raw = self.data.sensordata[s : s + SENSORS_PER_ROBOT]
        return (raw > 0.0).astype(np.float32)

    def _robot_cfrc_ext(self, i: int) -> np.ndarray:
        """External contact forces for all bodies of robot i. Shape: (78,)."""
        body_ids = self._robot_cfrc_body_ids[i]
        return self.data.cfrc_ext[body_ids].flatten().astype(np.float32)

    def _robot_pos(self, i: int) -> np.ndarray:
        """World-frame position of robot i's torso. Shape: (3,)."""
        return self.data.xpos[self._robot_body_ids[i]].copy()

    def _box_pos(self) -> np.ndarray:
        """World-frame position of the push box. Shape: (3,)."""
        return self.data.xpos[self._box_body_id].copy()

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """Return stacked observations: shape (n_robots, obs_dim_per_robot)."""
        torso_positions = np.stack([self._robot_pos(i) for i in range(self.n_robots)])

        obs_list: list[np.ndarray] = []
        for i in range(self.n_robots):
            qpos = self._robot_qpos(i)
            qvel = self._robot_qvel(i)

            # qpos layout: [x, y, z, qw, qx, qy, qz, j0..j11]
            # Include full qpos (exclude_current_positions_from_observation=False)
            parts: list[np.ndarray] = [
                qpos,                             # full qpos incl. x,y = 19
                qvel,                             # 6 free-joint vel + 12 joint vel = 18
                self._robot_foot_contacts(i),     # 4
            ]
            if self.include_cfrc_ext:
                parts.append(self._robot_cfrc_ext(i))  # 78
            proprio = np.concatenate(parts).astype(np.float32)

            # Relative positions of neighbors
            neighbor_parts: list[np.ndarray] = []
            for j in range(self.n_robots):
                if j != i:
                    neighbor_parts.append(
                        (torso_positions[j] - torso_positions[i]).astype(np.float32)
                    )
            neighbor_arr = (
                np.concatenate(neighbor_parts) if neighbor_parts else np.zeros(0, dtype=np.float32)
            )

            obs_list.append(
                np.concatenate([proprio, neighbor_arr, self._task_emb])
            )

        return np.stack(obs_list)  # (N, obs_dim)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def _is_healthy(self) -> bool:
        lo, hi = self.healthy_z_range
        for i in range(self.n_robots):
            z = self._robot_qpos(i)[2]
            if not (lo <= z <= hi):
                return False
        return bool(
            np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        )

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # "standing" keyframe

        # Add noise to initial joint angles / velocities
        noise_scale = self.reset_noise_scale
        for i in range(self.n_robots):
            qs = self._robot_qpos_starts[i]
            vs = self._robot_qvel_starts[i]
            self.data.qpos[qs : qs + ROBOT_QPOS_LEN] += self.np_random.uniform(
                -noise_scale, noise_scale, ROBOT_QPOS_LEN
            )
            self.data.qvel[vs : vs + ROBOT_QVEL_LEN] += self.np_random.uniform(
                -noise_scale, noise_scale, ROBOT_QVEL_LEN
            )

        self._reset_task_specific()
        mujoco.mj_forward(self.model, self.data)

        self._episode_return = 0.0
        self._episode_length = 0

        obs = self._build_obs()
        return {"obs": obs, "task_id": self.task_id}, {}

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        """Step all robots simultaneously.

        Args:
            actions: (n_robots, 12) float32 torques in [-1, 1].

        Returns:
            obs dict, team reward, terminated, truncated, info.
        """
        # Write control signals
        for i in range(self.n_robots):
            s = self._robot_act_starts[i]
            self.data.ctrl[s : s + ROBOT_ACTUATORS] = np.clip(actions[i], -1.0, 1.0)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._build_obs()
        reward, info = self._compute_reward(actions)

        # Add healthy (alive) bonus
        healthy = self._is_healthy()
        if healthy:
            reward += self.healthy_reward_value
        info["healthy_reward"] = self.healthy_reward_value if healthy else 0.0

        terminated = not healthy
        self._episode_length += 1
        truncated = self._episode_length >= self.max_episode_steps

        self._episode_return += reward

        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_return,
                "l": self._episode_length,
            }

        return {"obs": obs, "task_id": self.task_id}, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_reward(self, actions: np.ndarray) -> tuple[float, dict]:
        """Return (scalar_team_reward, info_dict)."""
        ...

    def _reset_task_specific(self) -> None:
        """Override for task-specific reset logic (e.g., place push box)."""
        # Default: move box far away so it doesn't interfere
        if self._box_body_id >= 0:
            bqs = self._box_qpos_start
            self.data.qpos[bqs : bqs + 3] = [100.0, 100.0, -10.0]
            self.data.qpos[bqs + 3 : bqs + 7] = [1.0, 0.0, 0.0, 0.0]
