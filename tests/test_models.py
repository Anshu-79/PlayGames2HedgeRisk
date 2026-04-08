# tests/test_models.py

import numpy as np
import pytest
from src.models.statistical.models import (
    HistoricalSimulation, ParametricVaR, CAViaR
)


@pytest.fixture
def fake_data():
    np.random.seed(42)
    X = np.random.randn(300, 5)
    y = np.random.normal(-0.001, 0.015, 300)
    return X, y


def test_historical_sim_fit_predict(fake_data):
    X, y = fake_data
    m = HistoricalSimulation(quantile=0.95)
    m.fit(X, y)
    var = m.predict_var(X[:10])
    cvar = m.predict_cvar(X[:10])
    assert var.shape == (10,)
    assert cvar.shape == (10,)
    assert (var >= cvar).all(), "CVaR should be <= VaR (more negative)"


def test_parametric_normal(fake_data):
    X, y = fake_data
    m = ParametricVaR(quantile=0.95, dist="normal")
    m.fit(X, y)
    var = m.predict_var(X[:5])
    assert var.shape == (5,)
    assert np.all(np.isfinite(var))


def test_parametric_student_t(fake_data):
    X, y = fake_data
    m = ParametricVaR(quantile=0.95, dist="student-t")
    m.fit(X, y)
    var = m.predict_var(X[:5])
    assert np.all(np.isfinite(var))


def test_caviar_fit_predict(fake_data):
    X, y = fake_data
    m = CAViaR(quantile=0.95, n_iter=200)
    m.fit(X, y)
    var = m.predict_var(X[:5])
    assert var.shape == (5,)
    assert np.all(np.isfinite(var))


def test_base_predict_dict(fake_data):
    X, y = fake_data
    m = HistoricalSimulation(quantile=0.95)
    m.fit(X, y)
    out = m.predict(X[:3])
    assert "var"  in out
    assert "cvar" in out
