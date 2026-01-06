"""
Part 4 helper script.

Usage:
    python run_alpaca_backtest.py --symbol AAPL --timeframe 1Min --limit 1000

Requires environment variables:
    ALPACA_API_KEY
    ALPACA_API_SECRET
Optional:
    ALPACA_API_URL (defaults to https://paper-api.alpaca.markets)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import alpaca_trade_api as tradeapi
import pandas as pd

from Strategy_Backtesting import Backtester, PerformanceAnalyzer, plot_equity
from data_pipeline import CLEAN_STOCK_DIR, clean_market_data
from gateway import MarketDataGateway
from matching_engine import MatchingEngine
from order_book import OrderBook
from order_manager import OrderLoggingGateway, OrderManager
from strategy_base import MovingAverageStrategy


def fetch_alpaca_bars(
    symbol: str,
    timeframe: str,
    limit: int,
    api_key: str,
    api_secret: str,
    base_url: str,
) -> pd.DataFrame:
    api = tradeapi.REST(api_key, api_secret, base_url)
    bars = api.get_bars(symbol, timeframe, limit=limit).df.reset_index()
    bars.rename(
        columns={
            "timestamp": "Datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    if "trade_count" in bars.columns:
        bars.drop(columns=["trade_count", "vwap"], errors="ignore", inplace=True)
    bars = bars[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
    bars["Datetime"] = pd.to_datetime(bars["Datetime"], utc=True)
    return bars


def save_bars(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    CLEAN_STOCK_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = CLEAN_STOCK_DIR / f"{symbol.upper()}_{timeframe}_alpaca_raw.csv"
    df.to_csv(raw_path, index=False)
    return raw_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Alpaca data and run backtest.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol to download.")
    parser.add_argument("--timeframe", default="1Min", help="Alpaca timeframe (e.g., 1Min, 5Min).")
    parser.add_argument("--limit", type=int, default=1000, help="Number of bars to fetch.")
    parser.add_argument("--short-window", type=int, default=20, help="Short MA window.")
    parser.add_argument("--long-window", type=int, default=60, help="Long MA window.")
    parser.add_argument("--position-size", type=int, default=10, help="Per-trade position size.")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital.")
    parser.add_argument("--plot", action="store_true", help="Plot equity curve at the end.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ["ALPACA_API_KEY"]
    api_secret = os.environ["ALPACA_API_SECRET"]
    base_url = os.environ.get("ALPACA_API_URL", "https://paper-api.alpaca.markets")

    print(f"Downloading {args.limit} {args.timeframe} bars for {args.symbol} from Alpaca...")
    raw_df = fetch_alpaca_bars(args.symbol, args.timeframe, args.limit, api_key, api_secret, base_url)
    raw_csv = save_bars(raw_df, args.symbol, args.timeframe)

    print("Cleaning data and adding derived features...")
    clean_csv = clean_market_data(raw_csv)

    print("Initializing backtester components...")
    gateway = MarketDataGateway(clean_csv)
    strategy = MovingAverageStrategy(
        short_window=args.short_window,
        long_window=args.long_window,
        position_size=args.position_size,
    )
    order_book = OrderBook()
    order_manager = OrderManager(
        capital=args.capital,
        max_long_position=5 * args.position_size,
        max_short_position=5 * args.position_size,
    )
    matching_engine = MatchingEngine()
    logger = OrderLoggingGateway("order_log.json")

    backtester = Backtester(
        data_gateway=gateway,
        strategy=strategy,
        order_manager=order_manager,
        order_book=order_book,
        matching_engine=matching_engine,
        logger=logger,
        default_position_size=args.position_size,
    )

    print("Running backtest...")
    equity_df = backtester.run()

    analyzer = PerformanceAnalyzer(equity_df["equity"].tolist(), backtester.trades)
    print("\n=== Performance ===")
    print(f"PnL: {analyzer.pnl():.2f}")
    print(f"Sharpe: {analyzer.sharpe():.2f}")
    print(f"Max Drawdown: {analyzer.max_drawdown():.4f}")
    print(f"Win Rate: {analyzer.win_rate():.2%}")
    print(f"Trades executed: {len([t for t in backtester.trades if t.qty > 0])}")

    if args.plot:
        plot_equity(equity_df)


if __name__ == "__main__":
    main()
