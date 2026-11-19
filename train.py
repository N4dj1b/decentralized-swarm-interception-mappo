from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.swarm_env import EnvConfig, SwarmInterceptionEnv
from src.mappo import MAPPOAgent, MAPPOConfig, train_mappo
from src.evaluation import evaluate_agent, evaluate_baseline


def main():
    parser = argparse.ArgumentParser(description="Train a toy decentralized MAPPO drone-swarm interceptor.")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-seed-start", type=int, default=900000)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = SwarmInterceptionEnv(EnvConfig(seed=args.seed))
    agent = MAPPOAgent(
        env.obs_dim, env.state_dim, env.n_target_actions,
        MAPPOConfig(seed=args.seed),
    )

    history, losses = train_mappo(env, agent, total_episodes=args.episodes, seed_start=args.seed * 1000)
    import torch
    torch.save(agent.model.state_dict(), out / "mappo_actor_critic.pt")
    learned = evaluate_agent(env, agent, episodes=args.eval_episodes, seed_start=args.eval_seed_start)
    baseline = evaluate_baseline(env, episodes=args.eval_episodes, seed_start=args.eval_seed_start)

    result = {"learned_mappo": learned, "assignment_baseline": baseline,
              "training_episodes": args.episodes, "seed": args.seed,
              "evaluation_seed_start": args.eval_seed_start,
              "evaluation_seed_end": args.eval_seed_start + args.eval_episodes - 1,
              "note": ("Toy 2D prototype. MAPPO learns decentralized target selection only; "
                       "low-level movement uses a deterministic local pursuit controller. "
                       "Results are specific to this fixed configuration and are not physical-drone benchmarks.")}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    np.savez(out / "training_history.npz",
             rewards=np.asarray([x["episode_reward"] for x in history]),
             captured=np.asarray([x["captured"] for x in history]),
             leaked=np.asarray([x["leaked"] for x in history]))

    rewards = np.asarray([x["episode_reward"] for x in history])
    window = min(30, len(rewards))
    rolling = np.convolve(rewards, np.ones(window) / window, mode="valid") if window else rewards
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.25, label="Episode reward")
    if len(rolling):
        plt.plot(np.arange(window - 1, window - 1 + len(rolling)), rolling, label=f"Moving average ({window})")
    plt.xlabel("Training episode")
    plt.ylabel("Team reward")
    plt.title("MAPPO training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "training_curve.png", dpi=160)
    plt.close()

    labels = ["Full-defense rate", "Intruder interception rate"]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, [learned["full_defense_rate"], learned["intruder_interception_rate"]], width, label="MAPPO")
    plt.bar(x + width / 2, [baseline["full_defense_rate"], baseline["intruder_interception_rate"]], width, label="Assignment baseline")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title("Evaluation comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "evaluation_comparison.png", dpi=160)
    plt.close()

    print("\nEvaluation summary")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
