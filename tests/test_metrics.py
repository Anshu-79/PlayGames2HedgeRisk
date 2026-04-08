# tests/test_metrics.py

import numpy as np
import pytest
from src.evaluation.metrics import (
    violation_ratio, kupiec_test, christoffersen_test,
    tail_mean_error, joint_var_cvar_score, evaluate_var, evaluate_cvar
)


@pytest.fixture
def sample_data():
    np.random.seed(0)
    returns  = np.random.normal(-0.001, 0.01, 500)
    var_pred = np.full(500, np.quantile(returns, 0.05))  # 95% VaR
    cvar_pred = np.full(500, returns[returns <= var_pred[0]].mean())
    return returns, var_pred, cvar_pred


def test_violation_ratio_approx_one(sample_data):
    returns, var_pred, _ = sample_data
    vr = violation_ratio(returns, var_pred, quantile=0.95)
    assert 0.5 < vr < 1.5, f"Violation ratio out of expected range: {vr}"


def test_kupiec_returns_dict(sample_data):
    returns, var_pred, _ = sample_data
    result = kupiec_test(returns, var_pred, quantile=0.95)
    assert "statistic" in result
    assert "p_value"   in result
    assert "violations" in result


def test_christoffersen_returns_dict(sample_data):
    returns, var_pred, _ = sample_data
    result = christoffersen_test(returns, var_pred)
    assert "statistic" in result
    assert "p_value"   in result


def test_tail_mean_error_nonnegative(sample_data):
    returns, var_pred, cvar_pred = sample_data
    err = tail_mean_error(returns, cvar_pred, var_pred)
    assert err >= 0 or np.isnan(err)


def test_joint_score_finite(sample_data):
    returns, var_pred, cvar_pred = sample_data
    score = joint_var_cvar_score(returns, var_pred, cvar_pred, quantile=0.95)
    assert np.isfinite(score)


def test_evaluate_var_keys(sample_data):
    returns, var_pred, _ = sample_data
    metrics = evaluate_var(returns, var_pred, quantile=0.95)
    for key in ["violation_ratio", "kupiec_p_value", "christoffersen_p_value"]:
        assert key in metrics, f"Missing key: {key}"


def test_evaluate_cvar_keys(sample_data):
    returns, var_pred, cvar_pred = sample_data
    metrics = evaluate_cvar(returns, var_pred, cvar_pred, quantile=0.95)
    for key in ["tail_mean_error", "joint_score"]:
        assert key in metrics, f"Missing key: {key}"
