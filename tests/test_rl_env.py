# tests/test_rl_env.py

import numpy as np
import pytest
from src.models.rl.agent import RiskEnv, RLVaRAgent


@pytest.fixture
def env():
    np.random.seed(0)
    returns  = np.random.normal(-0.001, 0.01, 300)
    features = np.random.randn(300, 5)
    return RiskEnv(returns, features, window=20, quantile=0.95)


def test_env_reset(env):
    obs = env.reset()
    assert obs.shape == (20 * 5,)
    assert not env.done


def test_env_step(env):
    env.reset()
    obs, reward, done, info = env.step(-0.02)
    assert isinstance(reward, float)
    assert "violation" in info
    assert "r_t" in info
    assert "var_pred" in info


def test_env_runs_to_completion(env):
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(-0.02)
        steps += 1
    assert steps > 0


def test_agent_placeholder_train(env):
    agent = RLVaRAgent(quantile=0.95)
    # Should not crash (placeholder random agent)
    agent.train(env, n_episodes=3)


def test_agent_evaluate(env):
    agent = RLVaRAgent(quantile=0.95)
    agent.train(env, n_episodes=2)
    out = agent.evaluate(env)
    assert "var_preds"  in out
    assert "actuals"    in out
    assert "violations" in out
