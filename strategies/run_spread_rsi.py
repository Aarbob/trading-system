import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from strategies.strategy_base import Strategy, TradeRecord
from core.backtester import Backtester, OrderBook, OrderManager, MatchingEngine, OrderLoggingGateway, plot_equity

# -----------------------------
# 1️⃣ Merge RSP & SPY
# -----------------------------
def load_market_data(rsp_path: str, spy_path: str, vix_path: str = None) -> pd.DataFrame:
    df_rsp = pd.read_csv(rsp_path, parse_dates=["Datetime"])
    df_spy = pd.read_csv(spy_path, parse_dates=["Datetime"])
    df = pd.merge(df_rsp, df_spy, on="Datetime", suffixes=("_RSP", "_SPY"))

    if vix_path:
        df_vix = pd.read_csv(vix_path, parse_dates=["Datetime"])
        df = pd.merge(df, df_vix, on="Datetime")
    
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df

# -----------------------------
# 2️⃣ Relative-Value RSI Strategy
# -----------------------------
class SpreadRSIStrategy(Strategy):
    """
    Trade RSP vs SPY based on RSI of the spread.
    Buy RSP / Short SPY when spread is oversold, and vice versa.
    Optional: Only trade in low-volatility regime (VIX filter).
    """
    def __init__(self, rsi_window=14, oversold=30, overbought=70, position_size=10, vix_threshold=None):
        self.rsi_window = rsi_window
        self.oversold = oversold
        self.overbought = overbought
        self.position_size = position_size
        self.vix_threshold = vix_threshold  # Only trade if VIX < threshold
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Spread
        df["spread"] = df["Close_RSP"] / df["Close_SPY"]

        # RSI of spread
        delta = df["spread"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.rsi_window, min_periods=1).mean()
        avg_loss = loss.rolling(self.rsi_window, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Optionally include VIX
        if "VIX_Close" in df.columns and self.vix_threshold is not None:
            df["low_vol"] = df["VIX_Close"] < self.vix_threshold
        else:
            df["low_vol"] = True
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        # Only trade if low-volatility
        mask = df["low_vol"]

        buy = mask & (df["RSI"] < self.oversold)  # Spread oversold → long RSP, short SPY
        sell = mask & (df["RSI"] > self.overbought)  # Spread overbought → short RSP, long SPY

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        # Position & target_qty
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = self.position_size

        # For simplicity, use RSP as main symbol in order submission
        df["limit_price"] = df["Close_RSP"]
        return df

# -----------------------------
# 3️⃣ Run backtest
# -----------------------------
def run_backtest():
    # Paths to your CSVs
    data_dir = Path("data")
    df = load_market_data(
        rsp_path=data_dir / "RSP.csv",
        spy_path=data_dir / "SPY.csv",
        vix_path=data_dir / "VIX.csv"  # optional
    )

    strategy = SpreadRSIStrategy(
        rsi_window=14,
        oversold=30,
        overbought=70,
        position_size=10,
        vix_threshold=20  # Only trade in low-vol regime
    )

    # Initialize backtester
    order_book = OrderBook()
    order_manager = OrderManager(capital=50_000, max_long_position=1000, max_short_position=1000)
    matching_engine = MatchingEngine()
    logger = OrderLoggingGateway()

    # We'll wrap DataFrame into a simple gateway
    class DFDataGateway:
        def __init__(self, df: pd.DataFrame):
            self.df = df
        def stream(self):
            for _, row in self.df.iterrows():
                yield row.to_dict()

    gateway = DFDataGateway(df)

    backtester = Backtester(
        data_gateway=gateway,
        strategy=strategy,
        order_manager=order_manager,
        order_book=order_book,
        matching_engine=matching_engine,
        logger=logger,
    )

    equity_df = backtester.run()
    analyzer = backtester  # reuse backtester for trade list

    # Print summary
    print("=== Backtest Summary ===")
    print("Equity data points:", len(equity_df))
    print("Trades executed:", len([t for t in backtester.trades if t.qty > 0]))
    print("Final portfolio value:", equity_df["equity"].iloc[-1])
    
    # Plot equity curve
    plot_equity(equity_df)

if __name__ == "__main__":
    run_backtest()
