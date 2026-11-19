# Project brief — Decentralized Swarm Interception with MAPPO

**Goal.** Build a first research prototype around the target-assignment problem in the ENSTA U2IS internship proposal: several interceptors must coordinate which intruder to engage while intruders approach a protected asset.

**Environment.** A simplified 2D arena with two interceptors and three intruders. Intruders follow direct, split/lane, and decoy/evasive scripted behaviours. Interceptors have limited sensing and receive local observations plus a one-hot message containing the teammate's previous target intention.

**Learning method.** A shared decentralized actor selects each interceptor's target. A centralized critic sees the global state during training, following the CTDE paradigm. MAPPO is implemented in PyTorch with a clipped PPO objective and GAE. A deterministic local pursuit controller handles low-level cardinal movement; learning is focused on target assignment in this first experiment.

**Baseline.** A receding-horizon one-to-one assignment policy chooses the allocation minimizing total Euclidean distance, with the same pursuit controller.

**Experiment.** 512 training episodes; evaluation on 1,000 held-out reset seeds.

| Metric | MAPPO | Distance-only assignment |
|---|---:|---:|
| Full-defense rate | 99.9% | 97.3% |
| Intruder interception rate | 99.97% | 99.10% |
| Mean intruders captured | 2.999 | 2.973 |
| Mean intruders leaked | 0.001 | 0.027 |
| Mean time to capture (steps) | 16.21 | 11.86 |

**Interpretation.** In this fixed toy configuration, the learned assignment policy intercepted all but one intruder across the held-out test episodes, while the improved one-to-one distance baseline was faster on average and also performed strongly, although its full-defense rate remained below MAPPO in this test. This is preliminary evidence for the simplified assignment problem, not a claim of generalization to realistic drones, larger swarms, or unseen dynamics.

**Current limitation.** MAPPO learns target selection only. Low-level pursuit is hand-coded. Flight dynamics, GPU/JAX vectorization, ROS 2/PX4 integration, attention-based variable-size assignment, and learned intruder self-play remain future work.
