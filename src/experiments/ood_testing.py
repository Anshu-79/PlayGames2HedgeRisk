# src/experiments/ood_testing.py

import numpy as np
import pandas as pd
import mlflow
from src.evaluation.metrics import evaluate_var, evaluate_cvar
from src.utils.seed import set_seed


def run(model, df: pd.DataFrame, config: dict) -> dict:
    """
    OOD experiment: train on stable period, test on each crisis period.

    df must have DatetimeIndex with 'returns' and feature columns.
    """
    set_seed(config["experiment"]["seed"])
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    feature_cols = [c for c in df.columns if c not in ["returns", "close",
                                                         "open", "high", "low", "volume"]]
    stable = config["data"]["stable_period"]
    crises = config["data"]["crisis_periods"]
    quantile = config["model"]["quantile"]

    # Train on stable period
    stable_df = df.loc[stable["start"]: stable["end"]]
    X_train = stable_df[feature_cols].values
    y_train = stable_df["returns"].values
    model.fit(X_train, y_train)

    results = {}
    with mlflow.start_run(run_name=f"ood_{config['model']['type']}"):
        mlflow.log_param("stable_start", stable["start"])
        mlflow.log_param("stable_end",   stable["end"])
        mlflow.log_param("quantile",     quantile)

        for crisis in crises:
            name = crisis["name"]
            crisis_df = df.loc[crisis["start"]: crisis["end"]]
            if len(crisis_df) == 0:
                print(f"[WARN] No data for crisis {name}")
                continue

            X_test = crisis_df[feature_cols].values
            y_test = crisis_df["returns"].values

            preds   = model.predict(X_test)
            var_p   = preds["var"]
            cvar_p  = preds["cvar"]

            var_m  = evaluate_var(y_test, var_p, quantile)
            cvar_m = evaluate_cvar(y_test, var_p, cvar_p, quantile) if cvar_p is not None else {}

            all_m = {**var_m, **cvar_m}
            for k, v in all_m.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    mlflow.log_metric(f"{name}_{k}", v)

            results[name] = all_m
            print(f"\n--- OOD: {name} (n={len(y_test)}) ---")
            for k, v in all_m.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return results
