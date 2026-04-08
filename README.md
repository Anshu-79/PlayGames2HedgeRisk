# risk-hedging-rl

RL-based VaR/CVaR benchmarking framework — NIFTY 50 (25 years, yfinance).

## Setup

```bash
git clone <repo>
cd risk-hedging-rl
pip install -r requirements.txt
pip install -e .
```

## Run experiments

```bash
# Rolling window backtest (single model)
python main.py rolling --config configs/experiments/rolling_window.yaml

# Full benchmark (all models)
python main.py benchmark --config configs/experiments/rolling_window.yaml --all-models

# OOD / crisis testing
python main.py ood --config configs/experiments/ood_testing.yaml

# RL ablation study
python main.py ablation --config configs/experiments/ablation_rl.yaml

# Stability (multiple seeds)
python main.py stability --config configs/experiments/rolling_window.yaml --seeds 1,2,3,4,5
```

## View MLflow UI

```bash
mlflow ui --backend-store-uri outputs/mlruns
```

## Project structure

```
risk-hedging-rl/
├── configs/              # YAML configs (data, models, experiments)
├── data/                 # Raw + processed data
├── src/
│   ├── data/             # yfinance download + feature engineering
│   ├── models/
│   │   ├── statistical/  # HistSim, Parametric, GARCH, CAViaR
│   │   ├── ml/           # SVR, QGB, GARCH-SVR, MDN
│   │   ├── dl/           # LSTM, AttentionLSTM, Transformer, JointTransformer
│   │   └── rl/           # RiskEnv + RLVaRAgent (placeholder)
│   ├── evaluation/       # VaR/CVaR metrics (Kupiec, Christoffersen, ES)
│   ├── experiments/      # Rolling window, regime, OOD, ablation, stability
│   ├── utils/            # Seed, config loader, logger, model factory
│   └── visualization/    # 5 research plots
├── outputs/              # Logs, metrics CSVs, MLflow runs, plots
├── tests/                # Unit tests
└── main.py               # CLI entry point
```

## Model registry

| Key | Model |
|-----|-------|
| `historical_sim` | Historical Simulation |
| `parametric_normal` | Parametric VaR (Gaussian) |
| `parametric_t` | Parametric VaR (Student-t) |
| `garch` | GARCH(1,1) |
| `caviar` | CAViaR quantile regression |
| `svr` | Support Vector Regression |
| `qgb` | Quantile Gradient Boosting |
| `garch_svr` | GARCH-SVR hybrid |
| `mdn` | Mixture Density Network |
| `lstm` | Quantile LSTM |
| `attention_lstm` | Attention-LSTM |
| `transformer` | Transformer Encoder |
| `joint_transformer` | Joint VaR+CVaR Transformer |
| `rl_agent` | RL VaR Agent (**placeholder**) |

## RL status

`src/models/rl/agent.py` contains:
- `RiskEnv` — fully wired Gym-style environment with CVaR-aware reward
- `RLVaRAgent` — policy + evaluate scaffolding; **training loop is a placeholder**

Next steps for RL:
1. Replace random action with proper policy gradient update
2. Add value network (actor-critic / PPO)
3. Implement replay buffer for off-policy (SAC)
4. CVaR-aware reward shaping (done in env, wire to loss)
