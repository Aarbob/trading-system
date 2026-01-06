import numpy as np
import pandas as pd


class Strategy:
    """
    Base Strategy interface for adding indicators and generating trading signals.
    """

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        return df


class MovingAverageStrategy(Strategy):
    """
    Moving average crossover strategy with explicitly defined entry/exit rules.
    """

    def __init__(self, short_window: int = 20, long_window: int = 60, position_size: int = 10):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["MA_short"] = df["Close"].rolling(self.short_window, min_periods=1).mean()
        df["MA_long"] = df["Close"].rolling(self.long_window, min_periods=1).mean()
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(self.long_window).std().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = (df["MA_short"].shift(1) <= df["MA_long"].shift(1)) & (df["MA_short"] > df["MA_long"])
        sell = (df["MA_short"].shift(1) >= df["MA_long"].shift(1)) & (df["MA_short"] < df["MA_long"])

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = (df["position"].abs() * self.position_size).astype(int)
        return df


class PennyInPennyOutStrategy(Strategy):
    """
    Market-making inspired Penny-In-Penny-Out strategy.

    The strategy quotes inside the spread with a penny improvement, widens its
    edge when volatility increases, and leans its fair value against current
    inventory (fade). It exposes explicit bid/ask quotes plus per-side sizes,
    so the backtester can submit both legs in the same timestep.
    """

    def __init__(
        self,
        tick_size: float = 0.01,
        base_edge: float = 0.01,
        edge_range: float = 0.05,
        edge_sensitivity: float = 2.5,
        fair_ema_span: int = 20,
        fair_median_window: int = 30,
        vol_lookback: int = 30,
        spread_vol_multiplier: float = 0.6,
        min_spread: float = 0.02,
        max_spread: float = 0.50,
        fade_strength: float = 0.02,
        inventory_soft_limit: int = 200,
        base_qty: int = 10,
        max_quote_offset: float = 0.50,
        vol_halt: float = 0.10,
    ):
        if tick_size <= 0:
            raise ValueError("tick_size must be positive.")
        self.tick_size = tick_size
        self.base_edge = base_edge
        self.edge_range = edge_range
        self.edge_sensitivity = edge_sensitivity
        self.fair_ema_span = fair_ema_span
        self.fair_median_window = fair_median_window
        self.vol_lookback = vol_lookback
        self.spread_vol_multiplier = spread_vol_multiplier
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.fade_strength = fade_strength
        self.inventory_soft_limit = max(1, inventory_soft_limit)
        self.base_qty = max(1, base_qty)
        self.max_quote_offset = max_quote_offset
        self.vol_halt = vol_halt
        self.current_position = 0

    # ------------------------------------------------------------------ utils

    def update_context(self, position: int) -> None:
        """
        Allows the backtester to inject the live net position so fade can lean.
        """
        self.current_position = int(position)

    def _round_to_tick(self, price: pd.Series) -> pd.Series:
        return (price / self.tick_size).round() * self.tick_size

    # ------------------------------------------------------------- indicators

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ema_fair"] = df["Close"].ewm(span=self.fair_ema_span, adjust=False, min_periods=1).mean()
        df["median_fair"] = df["Close"].rolling(self.fair_median_window, min_periods=1).median()
        df["fair_price"] = (df["ema_fair"] + df["median_fair"]) / 2

        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(self.vol_lookback, min_periods=2).std().fillna(0.0)

        # Use realized volatility to inflate/deflate an estimated spread.
        spread_est = df["Close"] * df["volatility"] * self.spread_vol_multiplier + self.min_spread
        df["spread_est"] = spread_est.clip(lower=self.min_spread, upper=self.max_spread)

        # Approximate activity as volatility-normalized volume changes.
        vol_ma = df["Volume"].rolling(self.vol_lookback, min_periods=1).mean()
        df["activity_level"] = vol_ma.pct_change().abs().fillna(0.0)
        return df

    # -------------------------------------------------------------- signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0  # default for compatibility with existing consumers

        fair_price = df["fair_price"].fillna(df["Close"])

        # Fade: lean fair value against our inventory.
        inv_pressure = abs(self.current_position) / self.inventory_soft_limit
        fade_shift = np.sign(self.current_position) * self.fade_strength * np.log1p(abs(self.current_position))
        fade_adjustment = fade_shift * df["Close"]
        fair_with_fade = fair_price - fade_adjustment

        # Auto-edge: widen with volatility/activity, shrink when calm.
        spread_pressure = (df["spread_est"] - self.min_spread) / max(self.max_spread - self.min_spread, 1e-6)
        activity = df["activity_level"].clip(upper=5.0)
        auto_edge = self.base_edge + self.edge_range * np.tanh(
            -4 * self.edge_sensitivity * activity + 2
        )
        # Blend spread-driven and activity-driven edges.
        edge = (0.6 * self.base_edge + 0.4 * auto_edge) + spread_pressure * self.edge_range
        edge = edge.clip(lower=self.tick_size, upper=min(self.max_spread / 2, self.max_quote_offset))

        bid_price = self._round_to_tick(fair_with_fade - edge - self.tick_size)
        ask_price = self._round_to_tick(fair_with_fade + edge + self.tick_size)

        bid_qty = int(max(1, self.base_qty * (1 + max(0, -self.current_position) / self.inventory_soft_limit)))
        ask_qty = int(max(1, self.base_qty * (1 + max(0, self.current_position) / self.inventory_soft_limit)))

        df["bid_price"] = bid_price
        df["ask_price"] = ask_price
        df["bid_qty"] = bid_qty
        df["ask_qty"] = ask_qty

        # Halt quoting both sides if volatility is extreme; only work off inventory.
        high_vol = df["volatility"] > self.vol_halt
        df["bid_active"] = ~high_vol & (self.current_position < self.inventory_soft_limit * 1.5)
        df["ask_active"] = ~high_vol & (self.current_position > -self.inventory_soft_limit * 1.5)

        return df
