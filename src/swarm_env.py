"""A lightweight 2D swarm-interception environment for MAPPO experiments.

This is an educational research prototype, not a flight-physics simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class EnvConfig:
    n_interceptors: int = 2
    n_intruders: int = 3
    max_steps: int = 70
    interceptor_speed: float = 0.045
    intruder_speed: float = 0.018
    sensor_range: float = 0.88
    capture_radius: float = 0.055
    protected_radius: float = 0.105
    seed: int = 7


class SwarmInterceptionEnv:
    """Two-team toy environment with local observations and shared team reward.

    Each interceptor chooses (1) an intruder to engage, or no target, and
    (2) one of four cardinal movements or stay. Intruders use scripted
    direct, split, and decoy/evasive behaviours.
    """

    # 0: up, 1: down, 2: right, 3: left, 4: stay
    MOVE_VECTORS = np.asarray(
        [[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.n_interceptors = self.config.n_interceptors
        self.n_intruders = self.config.n_intruders
        self.rng = np.random.default_rng(self.config.seed)
        self.step_count = 0
        self.interceptor_pos = np.zeros((self.n_interceptors, 2), dtype=np.float32)
        self.intruder_pos = np.zeros((self.n_intruders, 2), dtype=np.float32)
        self.intruder_alive = np.ones(self.n_intruders, dtype=np.float32)
        self.captured_at = np.full(self.n_intruders, -1, dtype=np.int32)
        self.leaked = np.zeros(self.n_intruders, dtype=np.float32)
        self.last_assignments = np.full(self.n_interceptors, self.n_intruders, dtype=np.int64)
        self.done = False
        self.asset = np.asarray([0.5, 0.5], dtype=np.float32)

        # obs = own position, teammate relative positions, communicated teammate
        # target intentions (one-hot per teammate), intruder relative positions,
        # and local visibility/alive flags.
        self.obs_dim = (2 + 2 * (self.n_interceptors - 1)
                        + (self.n_intruders + 1) * (self.n_interceptors - 1)
                        + 3 * self.n_intruders)
        # Global state is available to the centralized critic only.
        self.state_dim = 2 * self.n_interceptors + 3 * self.n_intruders
        self.n_target_actions = self.n_intruders + 1  # final index means no target
        self.n_move_actions = len(self.MOVE_VECTORS)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.done = False
        # Interceptors start on opposite sides of the protected asset.
        if self.n_interceptors == 2:
            self.interceptor_pos = np.asarray([[0.34, 0.50], [0.66, 0.50]], dtype=np.float32)
        else:
            angles = np.linspace(0, 2 * np.pi, self.n_interceptors, endpoint=False)
            self.interceptor_pos = np.stack(
                [0.5 + 0.16 * np.cos(angles), 0.5 + 0.16 * np.sin(angles)], axis=1
            ).astype(np.float32)

        # Spawn intruders on varied edges. The role is determined by index:
        # direct approach, split/lane approach, and decoy/evasive approach.
        positions = []
        sides = self.rng.permutation(4)
        for i in range(self.n_intruders):
            side = int(sides[i % 4])
            t = float(self.rng.uniform(0.12, 0.88))
            if side == 0:      # north
                p = [t, 0.96]
            elif side == 1:    # south
                p = [t, 0.04]
            elif side == 2:    # east
                p = [0.96, t]
            else:              # west
                p = [0.04, t]
            positions.append(p)
        self.intruder_pos = np.asarray(positions, dtype=np.float32)
        self.intruder_alive = np.ones(self.n_intruders, dtype=np.float32)
        self.captured_at = np.full(self.n_intruders, -1, dtype=np.int32)
        self.leaked = np.zeros(self.n_intruders, dtype=np.float32)
        self.last_assignments = np.full(self.n_interceptors, self.n_intruders, dtype=np.int64)
        return self._observations(), self._global_state(), self._target_masks()

    def _target_masks(self) -> np.ndarray:
        masks = np.zeros((self.n_interceptors, self.n_target_actions), dtype=bool)
        for i in range(self.n_interceptors):
            distances = np.linalg.norm(self.intruder_pos - self.interceptor_pos[i], axis=1)
            visible = (self.intruder_alive > 0) & (distances <= self.config.sensor_range)
            masks[i, : self.n_intruders] = visible
            masks[i, self.n_intruders] = True  # always permit no-target action
        return masks

    def _observations(self) -> np.ndarray:
        observations = []
        for i in range(self.n_interceptors):
            own = self.interceptor_pos[i]
            pieces = [own.copy()]
            for j in range(self.n_interceptors):
                if j != i:
                    pieces.append(self.interceptor_pos[j] - own)
                    intent = np.zeros(self.n_target_actions, dtype=np.float32)
                    intent[self.last_assignments[j]] = 1.0
                    pieces.append(intent)
            rel = self.intruder_pos - own
            distances = np.linalg.norm(rel, axis=1)
            visible = (self.intruder_alive > 0) & (distances <= self.config.sensor_range)
            rel = rel.copy()
            rel[~visible] = 0.0
            pieces.extend([rel.reshape(-1), visible.astype(np.float32)])
            observations.append(np.concatenate(pieces).astype(np.float32))
        return np.stack(observations)

    def _global_state(self) -> np.ndarray:
        return np.concatenate(
            [self.interceptor_pos.reshape(-1), self.intruder_pos.reshape(-1), self.intruder_alive]
        ).astype(np.float32)

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else np.zeros_like(vector)

    def _move_intruders(self):
        for k in range(self.n_intruders):
            if self.intruder_alive[k] <= 0:
                continue
            p = self.intruder_pos[k]
            to_asset = self._unit(self.asset - p)
            closest_i = int(np.argmin(np.linalg.norm(self.interceptor_pos - p, axis=1)))
            away = p - self.interceptor_pos[closest_i]
            threat_distance = float(np.linalg.norm(away))

            if k % 3 == 0:  # direct: shortest path to the protected asset
                direction = to_asset
            elif k % 3 == 1:  # split: approach from a designated lateral lane
                lane_targets = [
                    np.asarray([0.37, 0.50], dtype=np.float32),
                    np.asarray([0.63, 0.50], dtype=np.float32),
                    np.asarray([0.50, 0.37], dtype=np.float32),
                ]
                lane = lane_targets[k % len(lane_targets)]
                direction = self._unit(0.75 * (lane - p) + 0.25 * (self.asset - p))
            else:  # decoy: approach the asset, but sidestep a nearby interceptor
                phase = 0.20 * self.step_count + 0.7 * k
                lateral = np.asarray([-to_asset[1], to_asset[0]], dtype=np.float32)
                direction = self._unit(to_asset + 0.32 * np.sin(phase) * lateral)
                if threat_distance < 0.24:
                    direction = self._unit(0.60 * direction + 0.70 * self._unit(away))

            self.intruder_pos[k] = np.clip(
                p + self.config.intruder_speed * direction, 0.0, 1.0
            )

    def step(self, target_actions: np.ndarray, move_actions: np.ndarray):
        if self.done:
            raise RuntimeError("Episode is finished. Call reset() before step().")
        target_actions = np.asarray(target_actions, dtype=np.int64).reshape(self.n_interceptors)
        move_actions = np.asarray(move_actions, dtype=np.int64).reshape(self.n_interceptors)
        if np.any((target_actions < 0) | (target_actions >= self.n_target_actions)):
            raise ValueError("Invalid target action")
        if np.any((move_actions < 0) | (move_actions >= self.n_move_actions)):
            raise ValueError("Invalid movement action")

        alive_before = self.intruder_alive.copy()
        old_min_distances = np.min(
            np.linalg.norm(
                self.intruder_pos[:, None, :] - self.interceptor_pos[None, :, :], axis=2
            ),
            axis=1,
        )
        old_min_distances[alive_before <= 0] = 0.0
        assigned_old_distances = np.full(self.n_interceptors, np.nan, dtype=np.float32)
        for i, target in enumerate(target_actions):
            if target < self.n_intruders and alive_before[target] > 0:
                assigned_old_distances[i] = float(np.linalg.norm(self.interceptor_pos[i] - self.intruder_pos[target]))

        # Interceptor movement: the policy chooses its own target and movement.
        deltas = self.MOVE_VECTORS[move_actions] * self.config.interceptor_speed
        self.interceptor_pos = np.clip(self.interceptor_pos + deltas, 0.0, 1.0)

        # Scripted opposing swarm moves after interceptor actions.
        self.step_count += 1
        self._move_intruders()

        # Capture checks; a given intruder can only be captured once.
        captured_now = []
        for k in range(self.n_intruders):
            if self.intruder_alive[k] <= 0:
                continue
            distances = np.linalg.norm(self.interceptor_pos - self.intruder_pos[k], axis=1)
            if float(np.min(distances)) <= self.config.capture_radius:
                self.intruder_alive[k] = 0.0
                self.captured_at[k] = self.step_count
                captured_now.append(k)

        # Any surviving intruder entering the protected zone is a leak.
        leaked_now = []
        for k in range(self.n_intruders):
            if self.intruder_alive[k] > 0 and np.linalg.norm(self.intruder_pos[k] - self.asset) <= self.config.protected_radius:
                self.intruder_alive[k] = 0.0
                self.leaked[k] = 1.0
                leaked_now.append(k)

        new_min_distances = np.min(
            np.linalg.norm(
                self.intruder_pos[:, None, :] - self.interceptor_pos[None, :, :], axis=2
            ),
            axis=1,
        )
        new_min_distances[self.intruder_alive <= 0] = 0.0
        # Dense potential shaping helps the policy learn pursuit before captures occur.
        progress = float(np.sum((old_min_distances - new_min_distances) * alive_before))
        assigned_progress = []
        for i, target in enumerate(target_actions):
            if target < self.n_intruders and assigned_old_distances[i] == assigned_old_distances[i]:
                if self.intruder_alive[target] > 0:
                    new_assigned_distance = float(np.linalg.norm(self.interceptor_pos[i] - self.intruder_pos[target]))
                else:
                    new_assigned_distance = 0.0
                assigned_progress.append(float(assigned_old_distances[i]) - new_assigned_distance)
        assignment_shaping = float(np.mean(assigned_progress)) if assigned_progress else 0.0
        reward = (-0.01 + 0.10 * progress + 0.22 * assignment_shaping
                  + 1.35 * len(captured_now) - 2.0 * len(leaked_now))

        # Discourage both agents from committing to the same target simultaneously.
        active_assignments = [int(a) for a in target_actions if a < self.n_intruders and self.intruder_alive[a] > 0]
        if len(active_assignments) != len(set(active_assignments)):
            reward -= 0.04

        all_captured = bool(np.all(self.intruder_alive == 0) and np.sum(self.leaked) == 0)
        if all_captured:
            reward += 1.5
        time_limit = self.step_count >= self.config.max_steps
        self.done = all_captured or len(leaked_now) > 0 or time_limit

        self.last_assignments = target_actions.copy()
        info: dict[str, Any] = {
            "captured_now": captured_now,
            "leaked_now": leaked_now,
            "captured_count": int(np.sum(self.captured_at >= 0)),
            "leaked_count": int(np.sum(self.leaked)),
            "all_captured": all_captured,
            "steps": self.step_count,
        }
        return self._observations(), self._global_state(), self._target_masks(), float(reward), self.done, info

    def render(self, ax=None):
        """Draw the current 2D state on a Matplotlib axis."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.clear()
        circle = plt.Circle(self.asset, self.config.protected_radius, color="green", alpha=0.25)
        ax.add_patch(circle)
        ax.scatter(*self.asset, marker="*", s=220, label="Protected asset")
        alive = self.intruder_alive > 0
        if np.any(alive):
            ax.scatter(self.intruder_pos[alive, 0], self.intruder_pos[alive, 1], marker="x", s=90, label="Intruders")
        ax.scatter(self.interceptor_pos[:, 0], self.interceptor_pos[:, 1], marker="^", s=100, label="Interceptors")
        for i, p in enumerate(self.interceptor_pos):
            ax.text(p[0] + 0.012, p[1] + 0.012, f"I{i}")
        for k, p in enumerate(self.intruder_pos):
            if self.intruder_alive[k] > 0:
                ax.text(p[0] + 0.012, p[1] + 0.012, f"T{k}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", title=f"Swarm interception — step {self.step_count}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        return ax
