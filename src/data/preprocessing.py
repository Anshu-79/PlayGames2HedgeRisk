# src/data/preprocessing.py

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path


def download_nifty50(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download NIFTY 50 OHLCV data from yfinance."""
    print(f"Downloading {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = [c[0].lower() for c in df.columns]
    df.dropna(inplace=True)
    print(f"Downloaded {len(df)} rows.")
    return df


def compute_returns(df: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Add return column to dataframe."""
    if method == "log":
        df["returns"] = np.log(df["close"] / df["close"].shift(1))
    elif method == "simple":
        df["returns"] = df["close"].pct_change()
    else:
        raise ValueError(f"Unknown return method: {method}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistical features."""
    df["rolling_vol_20"] = df["returns"].rolling(20).std()
    df["rolling_vol_60"] = df["returns"].rolling(60).std()
    df["rolling_mean_20"] = df["returns"].rolling(20).mean()

    # Drawdown
    roll_max = df["close"].cummax()
    df["drawdown"] = (df["close"] - roll_max) / roll_max

    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split — no shuffling."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


def preprocess(config: dict) -> pd.DataFrame:
    """Full pipeline: download -> returns -> features -> save."""
    df = download_nifty50(
        ticker=config["ticker"],
        start=config["start_date"],
        end=config["end_date"],
    )
    df = compute_returns(df, method=config["returns"]["method"])
    df = add_features(df)
    df.dropna(inplace=True)

    out_path = Path(config["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"Saved processed data to {out_path}")
    return df


def load_processed(path: str) -> pd.DataFrame:
    """Load saved parquet."""
    return pd.read_parquet(path)
