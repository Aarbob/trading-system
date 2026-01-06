"""
Utilities for downloading, cleaning, and organizing market data for the
backtesting framework. Covers Part 1 of the project brief.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf


RAW_STOCK_DIR = Path("raw_data_stock")
RAW_CRYPTO_DIR = Path("raw_data_crypto")
CLEAN_STOCK_DIR = Path("clean_data_stock")
CLEAN_CRYPTO_DIR = Path("clean_data_crypto")

BINANCE_REST_ENDPOINT = "https://api.binance.com/api/v3/klines"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class DownloadResult:
    """Metadata returned after a download + clean cycle."""

    raw_path: Path
    clean_path: Path


def download_equity_data(
    ticker: str,
    period: str = "7d",
    interval: str = "1m",
    dest_dir: Path = RAW_STOCK_DIR,
) -> Path:
    """
    Download intraday equity data via yfinance and save to CSV.
    """
    dest_dir = _ensure_dir(dest_dir)
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")

    df.reset_index(inplace=True)
    df.rename(columns={"Datetime": "Datetime"}, inplace=True)
    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

    path = dest_dir / f"{ticker.upper()}_{interval}_raw.csv"
    df.to_csv(path, index=False)
    return path


def download_crypto_data(
    symbol: str,
    interval: str = "1m",
    limit: int = 1000,
    dest_dir: Path = RAW_CRYPTO_DIR,
) -> Path:
    """
    Download intraday cryptocurrency candles from Binance public REST API.
    """
    dest_dir = _ensure_dir(dest_dir)
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    response = requests.get(BINANCE_REST_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    candles = response.json()
    if not candles:
        raise ValueError(f"No data returned for symbol {symbol}.")

    records = []
    for candle in candles:
        open_time = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
        records.append(
            {
                "Datetime": open_time,
                "Open": float(candle[1]),
                "High": float(candle[2]),
                "Low": float(candle[3]),
                "Close": float(candle[4]),
                "Volume": float(candle[5]),
            }
        )

    df = pd.DataFrame(records)
    path = dest_dir / f"{symbol.upper()}_{interval}_raw.csv"
    df.to_csv(path, index=False)
    return path


def clean_market_data(
    csv_path: Path,
    dest_dir: Optional[Path] = None,
    add_features: bool = True,
) -> Path:
    """
    Clean downloaded data and optionally add derived features.
    """
    df = pd.read_csv(csv_path)
    if "Datetime" not in df.columns:
        raise ValueError("Input CSV must contain a Datetime column.")

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df.dropna(subset=["Datetime"], inplace=True)
    df.drop_duplicates(subset=["Datetime"], inplace=True)
    df.sort_values("Datetime", inplace=True)

    # restrict to required columns
    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
    df.set_index("Datetime", inplace=True)

    if add_features:
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["rolling_volatility"] = df["returns"].rolling(60).std().fillna(0.0)
        df["rolling_volume"] = df["Volume"].rolling(60).mean().fillna(method="bfill")
        df["momentum"] = df["Close"].diff().fillna(0.0)

    if dest_dir is None:
        parent = Path(csv_path).parent
        if parent == RAW_CRYPTO_DIR:
            dest_dir = CLEAN_CRYPTO_DIR
        elif parent == RAW_STOCK_DIR:
            dest_dir = CLEAN_STOCK_DIR
        else:
            dest_dir = Path("clean_data")

    dest_dir = _ensure_dir(dest_dir)
    out_path = dest_dir / f"{Path(csv_path).stem.replace('_raw', '')}_clean.csv"
    df.to_csv(out_path)
    return out_path


def run_data_pipeline(
    asset_type: str,
    symbol: str,
    period: str = "7d",
    interval: str = "1m",
) -> DownloadResult:
    """
    Convenience helper that downloads and cleans data in one call.
    """
    asset_type = asset_type.lower()
    if asset_type not in {"equity", "crypto"}:
        raise ValueError("asset_type must be 'equity' or 'crypto'")

    if asset_type == "equity":
        raw_path = download_equity_data(symbol, period=period, interval=interval)
        clean_path = clean_market_data(raw_path, dest_dir=CLEAN_STOCK_DIR)
    else:
        raw_path = download_crypto_data(symbol, interval=interval)
        clean_path = clean_market_data(raw_path, dest_dir=CLEAN_CRYPTO_DIR)

    return DownloadResult(raw_path=raw_path, clean_path=clean_path)


if __name__ == "__main__":
    # Example: download and clean 7 days of 1-minute AAPL data.
    result = run_data_pipeline(asset_type="equity", symbol="AAPL")
    print("Raw data saved to:", result.raw_path)
    print("Clean data saved to:", result.clean_path)
