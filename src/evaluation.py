"""Evaluation utilities and a simple one-to-one assignment baseline."""
from __future__ import annotations
import itertools
import numpy as np
from .mappo import decentralized_pursuit_guidance


def _movement_toward(delta: np.ndarray) -> int:
    dx, dy = float(delta[0]), float(delta[1])
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return 4
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 3
    return 0 if dy > 0 else 1


def assignment_baseline_actions(env):
    """Receding-horizon LAP-style baseline using exhaustive assignment (small M)."""
    active = [i for i, alive in enumerate(env.intruder_alive) if alive > 0]
    n = env.n_interceptors
    if not active:
        return np.full(n, env.n_intruders, dtype=np.int64), np.full(n, 4, dtype=np.int64)

    # Assign unique targets to as many interceptors as possible, minimizing
    # total distance. For this toy problem, exhaustive enumeration is enough.
    k = min(n, len(active))
    best_pairs, best_cost = None, float("inf")
    # Enumerate both the selected interceptors and their unique target
    # assignments. This remains exact when fewer intruders than interceptors
    # are alive; no interceptor is privileged by its array index.
    for interceptor_subset in itertools.combinations(range(n), k):
        for targets in itertools.permutations(active, k):
            cost = sum(
                float(np.linalg.norm(env.interceptor_pos[i] - env.intruder_pos[target]))
                for i, target in zip(interceptor_subset, targets)
            )
            if cost < best_cost:
                best_cost = cost
                best_pairs = tuple(zip(interceptor_subset, targets))
    target_actions = np.full(n, env.n_intruders, dtype=np.int64)
    move_actions = np.full(n, 4, dtype=np.int64)
    for interceptor, target in best_pairs:
        target_actions[interceptor] = int(target)
        move_actions[interceptor] = _movement_toward(env.intruder_pos[target] - env.interceptor_pos[interceptor])
    return target_actions, move_actions


def evaluate_agent(env, agent, episodes: int = 100, seed_start: int = 50000) -> dict:
    records = []
    for ep in range(episodes):
        obs, state, masks = env.reset(seed=seed_start + ep)
        steps = 0
        captured_at = []
        while True:
            target, _, _ = agent.act(obs, state, masks, deterministic=True)
            move = decentralized_pursuit_guidance(obs, target, env.n_interceptors, env.n_intruders)
            obs, state, masks, _, done, info = env.step(target, move)
            steps += 1
            captured_at.extend([steps] * len(info["captured_now"]))
            if done:
                records.append({
                    "steps": steps,
                    "captured": info["captured_count"],
                    "leaked": info["leaked_count"],
                    "all_captured": info["all_captured"],
                    "time_to_capture": float(np.mean(captured_at)) if captured_at else np.nan,
                })
                break
    return _summarize(records, episodes, env.n_intruders)


def evaluate_baseline(env, episodes: int = 100, seed_start: int = 50000) -> dict:
    records = []
    for ep in range(episodes):
        env.reset(seed=seed_start + ep)
        steps = 0
        captured_at = []
        while True:
            target, move = assignment_baseline_actions(env)
            _, _, _, _, done, info = env.step(target, move)
            steps += 1
            captured_at.extend([steps] * len(info["captured_now"]))
            if done:
                records.append({
                    "steps": steps,
                    "captured": info["captured_count"],
                    "leaked": info["leaked_count"],
                    "all_captured": info["all_captured"],
                    "time_to_capture": float(np.mean(captured_at)) if captured_at else np.nan,
                })
                break
    return _summarize(records, episodes, env.n_intruders)


def _summarize(records, episodes, n_intruders):
    captured = np.asarray([r["captured"] for r in records], dtype=float)
    leaked = np.asarray([r["leaked"] for r in records], dtype=float)
    times = np.asarray([r["time_to_capture"] for r in records], dtype=float)
    return {
        "episodes": int(episodes),
        "full_defense_rate": float(np.mean([r["all_captured"] for r in records])),
        "intruder_interception_rate": float(np.sum(captured) / (episodes * n_intruders)),
        "mean_captured_per_episode": float(np.mean(captured)),
        "mean_leaked_per_episode": float(np.mean(leaked)),
        "mean_time_to_capture": float(np.nanmean(times)) if np.any(np.isfinite(times)) else float("nan"),
        "mean_episode_steps": float(np.mean([r["steps"] for r in records])),
    }
