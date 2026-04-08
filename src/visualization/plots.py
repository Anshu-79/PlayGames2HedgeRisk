# src/visualization/plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid")
COLORS = sns.color_palette("tab10")


# ---------------------------------------------------------------------------
# Plot 1: Time Series — Returns + VaR + Violations
# ---------------------------------------------------------------------------

def plot_var_timeseries(
    dates: pd.DatetimeIndex,
    returns: np.ndarray,
    var_pred: np.ndarray,
    model_name: str = "Model",
    quantile: float = 0.95,
    save: bool = True,
) -> plt.Figure:
    """Plot actual returns vs VaR forecast with violations highlighted."""
    violations = returns < var_pred
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(dates, returns, color="steelblue", lw=0.8, alpha=0.7, label="Returns")
    ax.plot(dates, var_pred, color="firebrick", lw=1.5, linestyle="--",
            label=f"VaR ({int(quantile*100)}%)")
    ax.scatter(dates[violations], returns[violations],
               color="red", s=12, zorder=5, label=f"Violations (n={violations.sum()})")

    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_title(f"{model_name} — VaR Time Series (q={quantile})", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()

    if save:
        path = PLOT_DIR / f"var_timeseries_{model_name.lower().replace(' ','_')}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Tail Comparison — Histogram + CVaR estimates
# ---------------------------------------------------------------------------

def plot_tail_comparison(
    returns: np.ndarray,
    var_pred: float,
    cvar_pred: float,
    model_name: str = "Model",
    quantile: float = 0.95,
    save: bool = True,
) -> plt.Figure:
    """Histogram of actual losses with VaR and CVaR lines overlaid."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(returns, bins=80, color="steelblue", alpha=0.6, density=True, label="Actual Returns")

    ax.axvline(var_pred, color="darkorange", lw=2, linestyle="--",
               label=f"VaR ({int(quantile*100)}%) = {var_pred:.4f}")
    ax.axvline(cvar_pred, color="firebrick", lw=2, linestyle="-.",
               label=f"CVaR = {cvar_pred:.4f}")

    # Shade tail region
    tail_x = np.linspace(returns.min(), var_pred, 200)
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5],
                     returns.min(), var_pred,
                     color="red", alpha=0.08, label="Tail Region")

    ax.set_title(f"{model_name} — Tail Comparison", fontsize=13)
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    if save:
        path = PLOT_DIR / f"tail_comparison_{model_name.lower().replace(' ','_')}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 3: CVaR Error vs Time (rolling)
# ---------------------------------------------------------------------------

def plot_cvar_error_over_time(
    dates: pd.DatetimeIndex,
    actual_tail: np.ndarray,
    cvar_pred: np.ndarray,
    model_name: str = "Model",
    rolling_window: int = 30,
    save: bool = True,
) -> plt.Figure:
    """Rolling CVaR error over time — shows stability."""
    error = np.abs(cvar_pred - actual_tail)
    rolling_err = pd.Series(error).rolling(rolling_window).mean().values

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, error, color="lightcoral", lw=0.6, alpha=0.5, label="Point Error")
    ax.plot(dates, rolling_err, color="firebrick", lw=1.8,
            label=f"Rolling Mean (w={rolling_window})")

    ax.set_title(f"{model_name} — CVaR Error Over Time", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("|CVaR Error|")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()

    if save:
        path = PLOT_DIR / f"cvar_error_time_{model_name.lower().replace(' ','_')}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Regime Comparison Bar Plot
# ---------------------------------------------------------------------------

def plot_regime_comparison(
    regime_results: dict,
    metric: str = "tail_mean_error",
    save: bool = True,
) -> plt.Figure:
    """
    Bar chart: CVaR error per model in low vs high volatility regimes.

    regime_results: {model_name: {"low_vol": {...metrics}, "high_vol": {...metrics}}}
    """
    model_names = list(regime_results.keys())
    low_vals  = [regime_results[m].get("low_vol",  {}).get(metric, np.nan) for m in model_names]
    high_vals = [regime_results[m].get("high_vol", {}).get(metric, np.nan) for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, low_vals,  width, label="Low Vol",  color=COLORS[0], alpha=0.85)
    ax.bar(x + width/2, high_vals, width, label="High Vol", color=COLORS[3], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
    ax.set_title(f"Regime Comparison — {metric}", fontsize=13)
    ax.set_ylabel(metric)
    ax.legend()
    fig.tight_layout()

    if save:
        path = PLOT_DIR / f"regime_comparison_{metric}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 5: OOD Performance — Train vs Test gap
# ---------------------------------------------------------------------------

def plot_ood_gap(
    ood_results: dict,
    stable_metrics: dict,
    metric: str = "tail_mean_error",
    save: bool = True,
) -> plt.Figure:
    """
    Bar chart: model performance on stable (train) vs each crisis period.

    ood_results:    {crisis_name: {model: metric_value}}
    stable_metrics: {model: metric_value}
    """
    models   = list(stable_metrics.keys())
    periods  = ["stable"] + list(ood_results.keys())
    n_models = len(models)
    n_periods = len(periods)

    data = np.zeros((n_models, n_periods))
    for i, m in enumerate(models):
        data[i, 0] = stable_metrics.get(m, np.nan)
        for j, crisis in enumerate(list(ood_results.keys()), start=1):
            data[i, j] = ood_results[crisis].get(m, np.nan)

    x = np.arange(n_models)
    width = 0.8 / n_periods

    fig, ax = plt.subplots(figsize=(13, 5))
    for j, period in enumerate(periods):
        offset = (j - n_periods / 2) * width + width / 2
        ax.bar(x + offset, data[:, j], width, label=period,
               color=COLORS[j % len(COLORS)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_title(f"OOD Generalization Gap — {metric}", fontsize=13)
    ax.set_ylabel(metric)
    ax.legend(title="Period")
    fig.tight_layout()

    if save:
        path = PLOT_DIR / f"ood_gap_{metric}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig
