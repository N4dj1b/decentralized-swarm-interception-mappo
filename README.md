# Decentralized Swarm Interception with MAPPO

A compact, reproducible research prototype inspired by the ENSTA U2IS internship proposal **“Countering Coordinated Drone Swarms with Decentralized MARL.”** It focuses on one central subproblem from the proposal: **learning decentralized interceptor-to-intruder target assignment**.

> **Scope note:** this is a toy 2D research prototype, not a drone flight simulator. MAPPO learns target selection. A transparent local pursuit controller converts the selected target into a simple movement command. Low-level movement is therefore *not* learned in this version.

## Research question

Can a shared decentralized policy learn to assign two interceptors to three approaching intruders, using only local observations and a teammate's communicated target intention, and improve on a distance-only assignment baseline?

## What is implemented

- **Environment:** two interceptors defend a protected asset against three intruders in a normalized 2D arena. Interceptor and intruder motion use simplified kinematics.
- **Intruder behaviours:** direct approach, split/lane approach, and decoy/evasive approach. These are scripted opponent behaviours, as a first step toward the proposal's later competitive self-play objective.
- **Limited sensing:** each interceptor receives relative positions only for intruders inside its sensing radius.
- **Decentralized actor:** one shared actor is applied independently to each interceptor. Each actor receives its own position, relative teammate position, visible intruder-relative positions, visibility indicators, and a one-hot message representing the teammate's previous target intention.
- **Centralized critic:** during training, the critic receives the global state (both interceptor positions, all intruder positions, and intruder-alive flags). This is a simple **centralized training / decentralized execution (CTDE)** setup.
- **MAPPO:** clipped PPO objective, generalized advantage estimation (GAE), shared team reward, and a centralized value function.
- **Guidance:** after a target is selected, a deterministic local pursuit rule selects the cardinal movement that most reduces the relative target displacement. This isolates the learned assignment component.
- **Baseline:** a receding-horizon one-to-one assignment baseline. At every step it enumerates assignments and chooses the unique target allocation with the smallest total Euclidean distance, then uses the same pursuit controller. For this small problem, exhaustive enumeration gives the exact minimum-distance assignment without an external solver.
- **Evaluation:** full-defense rate, intruder interception rate, mean number captured/leaked, time to capture, and episode length.

## Experimental result

Configuration: 2 interceptors, 3 intruders, 512 MAPPO training episodes, seed 7. Evaluation used **1,000 held-out reset seeds** (`900000`–`900999), shared by the learned policy and the baseline.

| Metric | MAPPO learned assignment | Distance-only assignment baseline |
|---|---:|---:|
| Full-defense rate (all intruders intercepted) | **99.9%** | 97.3% |
| Intruder interception rate | **99.97%** | 99.10% |
| Mean intruders captured per episode | 2.999 | 2.973 |
| Mean intruders reaching protected area | 0.001 | 0.027 |
| Mean time to capture (steps) | 16.21 | 11.86 |
| Mean episode length (steps) | 25.94 | 18.48 |

The learned policy achieved a slightly higher full-defense and interception rate in this test configuration, but took longer on average to complete the captures. The baseline is a fairer receding-horizon one-to-one assignment that chooses both the interceptor subset and unique target allocation minimizing instantaneous total Euclidean distance. This remains a result from one simplified environment and one training configuration; it should **not** be interpreted as evidence of generalization to new dynamics, larger teams, physical drones, or unseen opponent strategies.

The exact evaluation output is in [`results/metrics.json`](results/metrics.json); the training and comparison plots are in [`results/`](results/). A one-page shareable summary is provided as [`PROJECT_BRIEF.pdf`](PROJECT_BRIEF.pdf).

## Run it

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py --episodes 512 --eval-episodes 1000 --seed 7 --output results
```

The run saves:

- `mappo_actor_critic.pt` — trained shared actor and centralized critic
- `metrics.json` — evaluation metrics
- `training_history.npz` — episode rewards, captures, and leaks
- `training_curve.png` — training reward curve
- `evaluation_comparison.png` — MAPPO vs. assignment-baseline comparison
- `trajectory_comparison.png` — representative held-out rollout paths

You can also open [`MAPPO_Swarm_Interception.ipynb`](MAPPO_Swarm_Interception.ipynb) and run the experiment step by step. The smoke tests can be run with `python -m pytest -q`.

## Project structure

```text
.
├── MAPPO_Swarm_Interception.ipynb
├── README.md
├── PROJECT_BRIEF.md
├── PROJECT_BRIEF.pdf
├── requirements.txt
├── train.py
├── src/
│   ├── __init__.py
│   ├── swarm_env.py
│   ├── mappo.py
│   └── evaluation.py
└── results/
    ├── metrics.json
    ├── training_history.npz
    ├── mappo_actor_critic.pt
    ├── training_curve.png
    ├── evaluation_comparison.png
    └── trajectory_comparison.png
```

## Limitations and next steps

This prototype is intentionally narrow. It does not implement continuous flight dynamics, acceleration/turn-rate constraints, collision avoidance, realistic sensing/noise, an attention/pointer network, GPU vectorization, JAX, ROS 2, or PX4. It also does not yet train the intruder team; the three intruders use scripted behaviours. The actor selects targets, while guidance remains deterministic.

The most relevant extensions toward the internship topic would be:

1. Replace the cardinal pursuit rule with a learned guidance action or continuous control policy.
2. Use an attention/pointer mechanism over a variable-sized set of observed intruders and teammate intentions.
3. Scale the number of interceptors/intruders and test `N < M` cases systematically.
4. Vectorize the simulator and compare PyTorch with a JAX implementation.
5. Replace scripted intruders with a second MAPPO team and study competitive self-play / opponent populations.
6. Validate the learned policy in a higher-fidelity simulator before making any real-world claims.

## Reproducibility and interpretation

All reported metrics were generated by the included code. The training seeds and held-out evaluation seeds are disjoint. The experiment is small and stochasticity comes primarily from randomized intruder spawn positions; intruder policies and movement rules are otherwise scripted and deterministic. Results may change with hyperparameters, initialization, or environment modifications.
