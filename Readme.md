# Trading System Project

Modular backtesting environment that follows the four parts outlined in the project
brief: data acquisition and preparation, exchange simulation, strategy
backtesting, and an Alpaca paper-trading handoff. Everything runs locally in
pure Python.

---

## Part 1 – Data Download and Preparation

### Step 1: Download Intraday Market Data
- `data_pipeline.download_equity_data()` pulls equity candles via `yfinance` and
  stores them in `raw_data_stock/` as CSV files with columns
  `Datetime, Open, High, Low, Close, Volume`.
- `data_pipeline.download_crypto_data()` fetches crypto candles from the Binance
  REST API and stores them in `raw_data_crypto/` with the same schema.

```python
from data_pipeline import download_equity_data, download_crypto_data

raw_equity_path = download_equity_data("AAPL", period="7d", interval="1m")
raw_crypto_path = download_crypto_data("BTCUSDT", interval="1m", limit=1000)
```

### Step 2: Clean and Organize Data
- `data_pipeline.clean_market_data()` removes missing/duplicate rows, enforces a
  chronological `Datetime` index, and adds derived features such as returns,
  rolling volatility, and momentum.
- The cleaned datasets are saved to `clean_data_stock/` or `clean_data_crypto/`
  and are ready for ingestion by the gateway.

```python
from data_pipeline import clean_market_data

clean_path = clean_market_data(raw_equity_path)
print("Cleaned data:", clean_path)
```

### Step 3: Create a Trading Strategy
- `strategy_base.Strategy` defines the interface, while
  `strategy_base.MovingAverageStrategy` implements a moving average crossover
  with clear entry/exit rules and position sizing logic.

```python
from strategy_base import MovingAverageStrategy

strategy = MovingAverageStrategy(short_window=20, long_window=60, position_size=15)
```

---

## Part 2 – Backtester Framework

### Market Data Gateway (`gateway.py`)
Reads cleaned CSV files and streams candles row-by-row through an iterator or
the `stream()` generator. Mimics a live feed by optionally adding delays.

### Order Book (`order_book.py`)
Stores bids and asks in heaps to enforce price-time priority. Supports order
addition, modification (by cancel + re-add), cancellation, and matching.

### Order Manager & Logging (`order_manager.py`)
Performs capital sufficiency checks, position-risk checks (long + short limits),
and order rate limiting. `OrderLoggingGateway` writes a JSONL audit trail of
submitted, rejected, and executed orders.

### Matching Engine (`matching_engine.py`)
Randomly decides whether matched orders are filled, partially filled, or
cancelled, and returns execution reports for the backtester.

---

## Part 3 – Strategy Backtesting

`Strategy_Backtesting.Backtester` ties together the gateway, strategy, order
book, order manager, matching engine, and logger to simulate execution. The
loop:
1. Pulls a new candle from the gateway.
2. Runs the strategy to update indicators and generate a signal.
3. Builds an order (with configurable position size) when signals fire.
4. Validates via `OrderManager` and inserts orders into the `OrderBook`.
5. Uses the `MatchingEngine` to simulate fills/partials/cancellations.
6. Tracks portfolio cash, positions, equity, and trade-level PnL.

`Strategy_Backtesting.PerformanceAnalyzer` computes PnL, Sharpe ratio, max
drawdown, and win rate. `plot_equity()` visualizes the equity curve.

**Quick start**
```
python test_system.py
```
This generates sample data (if needed), runs the backtest, and prints summary
statistics.

---

## Part 4 – Alpaca Trading Challenge
1. **Create an Alpaca account** at [alpaca.markets](https://alpaca.markets) and
   complete identity verification (paper trading only).
2. **Configure paper trading** via the Alpaca dashboard (Paper Overview → Reset
   virtual funds as needed).
3. **Obtain API keys** from the dashboard and keep them secure. Use only paper
   trading endpoints.
4. **Retrieve market data** with the Alpaca SDK:
   ```python
   import alpaca_trade_api as tradeapi

   api = tradeapi.REST(API_KEY, API_SECRET, "https://paper-api.alpaca.markets")
   data = api.get_bars("AAPL", "1Min", limit=100).df
   data.to_csv("alpaca_data/AAPL_1m.csv")
   ```
5. **Save market data** in flat files (CSV/Parquet) or a database. Organize by
   asset and timeframe, and handle timezones carefully.
6. **Use your Part 1 strategy** with Alpaca by feeding Alpaca candles through
   the same classes (`MarketDataGateway`, `Backtester`, etc.). No new strategy
   code is required; you can re-use and tune the existing implementation.

---

## Project Structure
```
clean_data_crypto/
clean_data_stock/
raw_data_crypto/
raw_data_stock/
data_pipeline.py
gateway.py
order_book.py
order_manager.py
matching_engine.py
strategy_base.py
Strategy_Backtesting.py
test_system.py
Readme.md
```

---

## Requirements
Create a virtual environment and install dependencies:
```
pip install -r requirements.txt
```
Key libraries: `pandas`, `numpy`, `matplotlib`, `yfinance`, `requests`,
`alpaca-trade-api` (for Part 4), and standard Python tooling.
