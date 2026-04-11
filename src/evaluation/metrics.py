# src/evaluation/metrics.py

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# VaR Metrics
# ---------------------------------------------------------------------------


def violation_ratio(
    y_true: np.ndarray, var_pred: np.ndarray, quantile: float = 0.95
) -> float:
    """
    Observed violation rate / expected violation rate.
    NOTE: This is NOT the raw violation rate.
    A value ~1 means correct coverage.
    """
    expected = 1 - quantile
    observed = np.mean(y_true < var_pred)
    return observed / expected if expected > 0 else np.nan


def kupiec_test(
    y_true: np.ndarray, var_pred: np.ndarray, quantile: float = 0.95
) -> dict:
    """
    Kupiec POF test.
    H0: violation rate == 1 - quantile
    Returns: test statistic, p-value.
    """
    q = 1 - quantile
    T = len(y_true)
    violations = np.sum(y_true < var_pred)
    p_hat = violations / T

    if violations == 0 or violations == T:
        return {"statistic": np.nan, "p_value": np.nan, "violations": violations}

    lr_stat = -2 * (
        violations * np.log(q / p_hat)
        + (T - violations) * np.log((1 - q) / (1 - p_hat))
    )
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    return {"statistic": lr_stat, "p_value": p_value, "violations": int(violations)}


def christoffersen_test(y_true: np.ndarray, var_pred: np.ndarray) -> dict:
    """
    Christoffersen independence test.
    Tests whether violations cluster in time.
    H0: violations are iid.
    Returns: test statistic, p-value.
    """
    hits = (y_true < var_pred).astype(int)
    T = len(hits)

    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        prev, curr = hits[t - 1], hits[t]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(x):
        return np.log(x + 1e-12)  # improved numerical stability

    L_ind = (
        safe_log(1 - pi_01) * n00
        + safe_log(pi_01) * n01
        + safe_log(1 - pi_11) * n10
        + safe_log(pi_11) * n11
    )
    L_iid = safe_log(1 - pi) * (n00 + n10) + safe_log(pi) * (n01 + n11)

    lr_stat = -2 * (L_iid - L_ind)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    return {"statistic": lr_stat, "p_value": p_value}


def evaluate_var(
    y_true: np.ndarray, var_pred: np.ndarray, quantile: float = 0.95
) -> dict:
    """Aggregate all VaR metrics."""
    return {
        "violation_ratio": violation_ratio(y_true, var_pred, quantile),
        **{
            f"kupiec_{k}": v for k, v in kupiec_test(y_true, var_pred, quantile).items()
        },
        **{
            f"christoffersen_{k}": v
            for k, v in christoffersen_test(y_true, var_pred).items()
        },
    }


# ---------------------------------------------------------------------------
# CVaR Metrics
# ---------------------------------------------------------------------------


def tail_mean_error(
    y_true: np.ndarray, cvar_pred: np.ndarray, var_pred: np.ndarray
) -> float:
    """Mean absolute error between predicted CVaR and actual tail mean."""
    tail_mask = y_true < var_pred
    if tail_mask.sum() == 0:
        return np.nan
    actual_tail_mean = y_true[tail_mask].mean()
    predicted_cvar_mean = cvar_pred[tail_mask].mean()
    return abs(predicted_cvar_mean - actual_tail_mean)


def joint_var_cvar_score(
    y_true: np.ndarray,
    var_pred: np.ndarray,
    cvar_pred: np.ndarray,
    quantile: float = 0.95,
) -> float:
    """
    Joint VaR-CVaR scoring rule.
    = quantile_loss(VaR) + tail_penalty(CVaR)
    """
    q = 1 - quantile

    # Correct pinball loss for VaR
    err_var = y_true - var_pred
    q_loss = np.mean(np.where(err_var < 0, (q - 1) * err_var, q * err_var))

    # ES penalty on tail
    tail = y_true < var_pred
    if tail.sum() > 0:
        es_penalty = np.mean((cvar_pred[tail] - y_true[tail]) ** 2)
    else:
        es_penalty = 0.0

    return q_loss + es_penalty


def es_backtest(
    y_true: np.ndarray, var_pred: np.ndarray, cvar_pred: np.ndarray
) -> dict:
    """
    Simple ES backtest: mean excess loss beyond VaR vs predicted CVaR.
    """
    violations = y_true < var_pred
    if violations.sum() == 0:
        return {"es_mean_error": np.nan, "n_violations": 0}
    excess = y_true[violations] - cvar_pred[violations]
    return {
        "es_mean_error": float(np.mean(excess)),
        "n_violations": int(violations.sum()),
    }


def evaluate_cvar(
    y_true: np.ndarray,
    var_pred: np.ndarray,
    cvar_pred: np.ndarray,
    quantile: float = 0.95,
) -> dict:
    """Aggregate all CVaR metrics."""
    return {
        "tail_mean_error": tail_mean_error(y_true, cvar_pred, var_pred),
        "joint_score": joint_var_cvar_score(y_true, var_pred, cvar_pred, quantile),
        **{f"es_{k}": v for k, v in es_backtest(y_true, var_pred, cvar_pred).items()},
    }
