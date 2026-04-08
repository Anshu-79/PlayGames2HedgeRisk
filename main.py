#!/usr/bin/env python
# main.py

import click
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import get_logger, get_git_hash
from src.utils.model_factory import build_model
from src.data.preprocessing import preprocess, load_processed


logger = get_logger("main")


def _load_data(data_cfg: dict):
    """Load or download+process NIFTY 50 data."""
    import os
    path = data_cfg["processed_path"]
    if os.path.exists(path):
        logger.info(f"Loading processed data from {path}")
        df = load_processed(path)
    else:
        logger.info("Processed data not found — downloading from yfinance...")
        df = preprocess(data_cfg)

    feature_cols = [c for c in df.columns if c not in
                    ["returns", "close", "open", "high", "low", "volume"]]
    returns  = df["returns"].values
    features = df[feature_cols].values
    return df, returns, features


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """NIFTY 50 VaR/CVaR Benchmarking Suite."""
    pass


@cli.command()
@click.option("--config", "-c", required=True, help="Path to experiment YAML config")
def rolling(config):
    """Run rolling window backtesting experiment."""
    from src.experiments.rolling_window import run as run_rolling

    cfg = load_config(config)
    set_seed(cfg["experiment"]["seed"])
    logger.info(f"Experiment: rolling_window | git: {get_git_hash()}")

    data_cfg = load_config(cfg["data"]["config"])
    df, returns, features = _load_data(data_cfg)
    model = build_model(cfg)

    logger.info(f"Model: {model}")
    run_rolling(model, returns, features, cfg)


@cli.command()
@click.option("--config", "-c", required=True, help="Path to experiment YAML config")
def ood(config):
    """Run OOD / crisis testing experiment."""
    from src.experiments.ood_testing import run as run_ood

    cfg = load_config(config)
    set_seed(cfg["experiment"]["seed"])
    logger.info(f"Experiment: ood_testing | git: {get_git_hash()}")

    data_cfg = load_config(cfg["data"]["config"])
    df, _, _ = _load_data(data_cfg)
    model = build_model(cfg)

    run_ood(model, df, cfg)


@cli.command()
@click.option("--config", "-c", required=True, help="Path to ablation YAML config")
def ablation(config):
    """Run RL ablation study."""
    from src.experiments.ablation_stability import run_ablation

    cfg = load_config(config)
    set_seed(cfg["experiment"]["seed"])
    logger.info(f"Experiment: ablation | git: {get_git_hash()}")

    data_cfg = load_config(cfg["data"]["config"])
    _, returns, features = _load_data(data_cfg)

    run_ablation(returns, features, cfg)


@cli.command()
@click.option("--config", "-c", required=True)
@click.option("--seeds", default="1,2,3,4,5", show_default=True,
              help="Comma-separated list of seeds")
def stability(config, seeds):
    """Run stability analysis over multiple seeds."""
    from src.experiments.ablation_stability import run_stability

    cfg    = load_config(config)
    seeds_ = [int(s) for s in seeds.split(",")]
    logger.info(f"Experiment: stability | seeds={seeds_} | git: {get_git_hash()}")

    data_cfg = load_config(cfg["data"]["config"])
    _, returns, features = _load_data(data_cfg)

    model_cls = build_model(cfg).__class__
    run_stability(model_cls, {"quantile": cfg["model"]["quantile"]},
                  returns, features, seeds_, cfg)


@cli.command()
@click.option("--config", "-c", required=True)
@click.option("--all-models", is_flag=True, default=False,
              help="Run all registered models sequentially")
def benchmark(config, all_models):
    """
    Run rolling window benchmark across ALL models.
    Useful for generating Table 1 / Table 2.
    """
    from src.experiments.rolling_window import run as run_rolling
    from src.utils.model_factory import MODEL_REGISTRY

    cfg      = load_config(config)
    data_cfg = load_config(cfg["data"]["config"])
    _, returns, features = _load_data(data_cfg)

    models_to_run = list(MODEL_REGISTRY.keys()) if all_models else [cfg["model"]["type"]]
    summary = {}

    for model_type in models_to_run:
        cfg["model"]["type"] = model_type
        logger.info(f"Benchmarking: {model_type}")
        try:
            model = build_model(cfg)
            result = run_rolling(model, returns, features, cfg)
            summary[model_type] = result["metrics"]
        except Exception as e:
            logger.warning(f"Model {model_type} failed: {e}")

    # Print summary table
    print("\n\n========== BENCHMARK SUMMARY ==========")
    rows = []
    for m, metrics in summary.items():
        row = {"model": m}
        row.update({k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metrics.items()})
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    cli()
