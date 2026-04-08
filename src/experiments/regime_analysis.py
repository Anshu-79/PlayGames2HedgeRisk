# src/experiments/regime_analysis.py

import numpy as np
import pandas as pd
import mlflow
from src.evaluation.metrics import evaluate_var, evaluate_cvar
from src.utils.seed import set_seed


def label_regimes(returns: np.ndarray, window: int = 60,
                  threshold: float = None) -> np.ndarray:
    """
    Label each timestep as 'low' or 'high' volatility.
    Threshold: median rolling vol by default.
    Returns array of 0 (low) / 1 (high).
    """
    roll_vol = pd.Series(returns).rolling(window).std().fillna(method="bfill").values
    if threshold is None:
        threshold = np.median(roll_vol)
    return (roll_vol > threshold).astype(int)


def run(model, returns: np.ndarray, features: np.ndarray,
        var_preds: np.ndarray, cvar_preds: np.ndarray,
        quantile: float = 0.95, config: dict = None) -> dict:
    """
    Evaluate pre-computed predictions split by volatility regime.
    Expects var_preds / cvar_preds aligned with returns.
    """
    set_seed(config["experiment"]["seed"])
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    regimes = label_regimes(returns)

    results = {}
    with mlflow.start_run(run_name=f"regime_{config['model']['type']}"):
        for label, name in [(0, "low_vol"), (1, "high_vol")]:
            mask = regimes == label
            if mask.sum() < 10:
                continue

            y = returns[mask]
            vp = var_preds[mask]
            cp = cvar_preds[mask] if cvar_preds is not None else None

            var_m = evaluate_var(y, vp, quantile)
            cvar_m = evaluate_cvar(y, vp, cp, quantile) if cp is not None else {}

            for k, v in {**var_m, **cvar_m}.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    mlflow.log_metric(f"{name}_{k}", v)

            results[name] = {**var_m, **cvar_m}
            print(f"\n--- {name} (n={mask.sum()}) ---")
            for k, v in results[name].items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return results
