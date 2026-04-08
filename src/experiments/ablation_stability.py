# src/experiments/ablation_stability.py

import numpy as np
import mlflow
from src.models.rl.agent import RLVaRAgent, RiskEnv
from src.evaluation.metrics import evaluate_var, evaluate_cvar
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Ablation Study (RL only)
# ---------------------------------------------------------------------------

def run_ablation(returns: np.ndarray, features: np.ndarray, config: dict) -> dict:
    """
    Systematically disable RL components and measure performance drop.
    Variants defined in config['ablation_variants'].
    """
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    quantile = config["model"]["quantile"]
    results  = {}

    for variant in config["ablation_variants"]:
        name  = variant["name"]
        set_seed(config["experiment"]["seed"])

        agent = RLVaRAgent(
            quantile=quantile,
            hidden_dim=config["model"]["hidden_dim"],
            cvar_reward=variant["cvar_reward"],
            risk_penalty=variant["risk_penalty"],
        )

        env = RiskEnv(
            returns=returns,
            features=features,
            quantile=quantile,
            cvar_penalty=1.0 if variant["cvar_reward"] else 0.0,
            risk_penalty=1.0 if variant["risk_penalty"] else 0.0,
        )

        with mlflow.start_run(run_name=f"ablation_{name}"):
            mlflow.log_params({
                "variant":      name,
                "cvar_reward":  variant["cvar_reward"],
                "risk_penalty": variant["risk_penalty"],
            })

            agent.train(env, n_episodes=50)     # placeholder train
            eval_out = agent.evaluate(env)

            var_m = evaluate_var(eval_out["actuals"], eval_out["var_preds"], quantile)
            for k, v in var_m.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    mlflow.log_metric(k, v)

            results[name] = var_m
            print(f"\n[Ablation] {name}: violation_ratio={var_m.get('violation_ratio', '?'):.4f}")

    return results


# ---------------------------------------------------------------------------
# Stability Analysis (multiple seeds)
# ---------------------------------------------------------------------------

def run_stability(model_cls, model_kwargs: dict,
                  returns: np.ndarray, features: np.ndarray,
                  seeds: list, config: dict) -> dict:
    """
    Train same model with multiple seeds; report mean ± std of metrics.
    """
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    from src.experiments.rolling_window import rolling_backtest
    quantile = config["model"]["quantile"]
    window   = config["data"]["window_size"]

    all_metrics = []

    for seed in seeds:
        set_seed(seed)
        model = model_cls(**model_kwargs)

        df = rolling_backtest(model, returns, features, window, quantile)
        df_clean = df.dropna(subset=["var_pred"])

        from src.evaluation.metrics import evaluate_var, evaluate_cvar
        m = evaluate_var(df_clean["actual"].values, df_clean["var_pred"].values, quantile)
        if df_clean["cvar_pred"].notna().all():
            m.update(evaluate_cvar(df_clean["actual"].values,
                                   df_clean["var_pred"].values,
                                   df_clean["cvar_pred"].values,
                                   quantile))
        all_metrics.append(m)

    # Aggregate
    keys = all_metrics[0].keys()
    summary = {}
    with mlflow.start_run(run_name=f"stability_{model_cls.__name__}"):
        for k in keys:
            vals = [m[k] for m in all_metrics
                    if m[k] is not None and not (isinstance(m[k], float) and np.isnan(m[k]))]
            if vals:
                mu, std = float(np.mean(vals)), float(np.std(vals))
                summary[k] = {"mean": mu, "std": std}
                mlflow.log_metric(f"{k}_mean", mu)
                mlflow.log_metric(f"{k}_std",  std)
                print(f"  {k}: {mu:.4f} ± {std:.4f}")

    return summary
