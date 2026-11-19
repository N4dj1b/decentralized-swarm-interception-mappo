import numpy as np

from src.swarm_env import EnvConfig, SwarmInterceptionEnv
from src.mappo import MAPPOAgent, MAPPOConfig, decentralized_pursuit_guidance
from src.evaluation import assignment_baseline_actions


def test_environment_and_policy_shapes():
    env = SwarmInterceptionEnv(EnvConfig(seed=11))
    obs, state, masks = env.reset(seed=11)
    assert obs.shape == (env.n_interceptors, env.obs_dim)
    assert state.shape == (env.state_dim,)
    assert masks.shape == (env.n_interceptors, env.n_target_actions)
    assert masks[:, -1].all()

    agent = MAPPOAgent(env.obs_dim, env.state_dim, env.n_target_actions, MAPPOConfig(seed=11))
    targets, _, _ = agent.act(obs, state, masks, deterministic=True)
    moves = decentralized_pursuit_guidance(obs, targets, env.n_interceptors, env.n_intruders)
    assert targets.shape == (env.n_interceptors,)
    assert moves.shape == (env.n_interceptors,)
    assert np.all((moves >= 0) & (moves < env.n_move_actions))


def test_assignment_baseline_produces_valid_actions():
    env = SwarmInterceptionEnv()
    env.reset(seed=3)
    targets, moves = assignment_baseline_actions(env)
    assert targets.shape == (env.n_interceptors,)
    assert moves.shape == (env.n_interceptors,)
    assert np.all((targets >= 0) & (targets < env.n_target_actions))
    assert np.all((moves >= 0) & (moves < env.n_move_actions))
