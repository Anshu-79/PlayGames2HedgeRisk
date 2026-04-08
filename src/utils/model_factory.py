# src/utils/model_factory.py

from src.models.statistical.models import (
    HistoricalSimulation, ParametricVaR, GARCHModel, CAViaR
)
from src.models.ml.models import (
    SVRModel, QuantileGradientBoosting, GARCHSVRModel, MixtureDensityNetwork
)
from src.models.dl.models import (
    LSTMQuantile, AttentionLSTM, TransformerEncoder, JointTransformer
)
from src.models.rl.agent import RLVaRAgent


MODEL_REGISTRY = {
    # Statistical
    "historical_sim":   HistoricalSimulation,
    "parametric_normal": lambda q: ParametricVaR(q, dist="normal"),
    "parametric_t":     lambda q: ParametricVaR(q, dist="student-t"),
    "garch":            GARCHModel,
    "caviar":           CAViaR,

    # ML
    "svr":              SVRModel,
    "qgb":              QuantileGradientBoosting,
    "garch_svr":        GARCHSVRModel,
    "mdn":              MixtureDensityNetwork,

    # DL
    "lstm":             LSTMQuantile,
    "attention_lstm":   AttentionLSTM,
    "transformer":      TransformerEncoder,
    "joint_transformer": JointTransformer,

    # RL
    "rl_agent":         RLVaRAgent,
}


def build_model(config: dict):
    model_type = config["model"]["type"]
    quantile   = config["model"]["quantile"]

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[model_type]

    # Lambda entries (parametric variants) need special handling
    if callable(cls) and not isinstance(cls, type):
        return cls(quantile)

    return cls(quantile=quantile)
