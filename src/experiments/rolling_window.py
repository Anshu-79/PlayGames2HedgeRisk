# src/experiments/rolling_window.py

import numpy as np
import pandas as pd
import mlflow
from tqdm import tqdm
from src.evaluation.metrics import evaluate_var, evaluate_cvar
from src.utils.seed import set_seed


def rolling_backtest(
    model,
    returns: np.ndarray,
    features: np.ndarray,
    window_size: int = 252,
    quantile: float = 0.95,
    config: dict = None,
) -> pd.DataFrame:
    """
    Rolling window backtesting.
    Trains model on [t-window:t], predicts at t, slides forward.

    Returns: DataFrame with columns [t, var_pred, cvar_pred, actual]
    """
    results = []
    T = len(returns)

    for t in tqdm(range(window_size, T), desc=f"Rolling backtest [{model}]"):
        X_train = features[t - window_size: t]
        y_train = returns[t - window_size: t]
        X_test  = features[t: t + 1]
        y_test  = returns[t]

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            var_t  = float(preds["var"][0]) if preds["var"] is not None else np.nan
            cvar_t = float(preds["cvar"][0]) if preds["cvar"] is not None else np.nan
        except Exception as e:
            print(f"[WARN] t={t}: {e}")
            var_t = cvar_t = np.nan

        results.append({
            "t": t,
            "var_pred":  var_t,
            "cvar_pred": cvar_t,
            "actual":    y_test,
        })

    df = pd.DataFrame(results)
    return df


def run(model, returns: np.ndarray, features: np.ndarray, config: dict) -> dict:
    """
    Full rolling window experiment with MLflow logging.
    """
    set_seed(config["experiment"]["seed"])

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    window    = config["data"]["window_size"]
    quantile  = config["model"]["quantile"]
    model_tag = config["model"]["type"]

    with mlflow.start_run(run_name=f"{model_tag}_q{quantile}"):
        mlflow.log_params({
            "model":       model_tag,
            "quantile":    quantile,
            "window_size": window,
            "seed":        config["experiment"]["seed"],
        })

        # Run backtest
        df = rolling_backtest(model, returns, features, window, quantile, config)

        # Drop NaN rows
        df_clean = df.dropna(subset=["var_pred"])

        # Evaluate
        var_metrics  = evaluate_var(
            df_clean["actual"].values,
            df_clean["var_pred"].values,
            quantile,
        )

        cvar_metrics = {}
        if df_clean["cvar_pred"].notna().all():
            cvar_metrics = evaluate_cvar(
                df_clean["actual"].values,
                df_clean["var_pred"].values,
                df_clean["cvar_pred"].values,
                quantile,
            )

        all_metrics = {**var_metrics, **cvar_metrics}
        mlflow.log_metrics({k: v for k, v in all_metrics.items()
                            if v is not None and not np.isnan(v)})

        # Save results CSV
        out_path = f"outputs/metrics/rolling_{model_tag}_q{int(quantile*100)}.csv"
        df.to_csv(out_path, index=False)
        mlflow.log_artifact(out_path)

        print(f"\n=== {model_tag} @ q={quantile} ===")
        for k, v in all_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return {"results_df": df, "metrics": all_metrics}
