import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from backtesting import (
    fetch_bars,
    calculate_metrics,
    run_backtest,
    strategy,
)
import subprocess
import sys


def test_strategy_buy_and_sell():
    # BUY
    pnl = strategy("BUY", "2024-01-01", 100, 110, 1000)
    assert pnl == (110 - 100) * (1000 // 100)
    # SELL
    pnl = strategy("SELL", "2024-01-01", 100, 90, 1000)
    assert isinstance(pnl, float)
    # HOLD
    pnl = strategy("HOLD", "2024-01-01", 100, 110, 1000)
    assert pnl == 0


def test_calculate_metrics():
    idx = pd.date_range("2024-01-01", periods=5)
    results_df = pd.DataFrame({"value": [100, 110, 120, 130, 140]}, index=idx)
    daily_returns = [{"date": d, "value": 0.01} for d in idx]
    cr, arr, sr, mdd = calculate_metrics(results_df, daily_returns)
    assert cr > 0
    assert arr > 0
    assert sr >= 0
    assert mdd <= 0


def test_run_backtest_single_worker():
    # Mock agent with propagate method
    class DummyAgent:
        def propagate(self, symbol, trade_date_str):
            return None, "BUY"

    bars_df = pd.DataFrame(
        {"open": [100, 105], "close": [110, 100]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    initial_cash = 1000
    portfolio_value, daily_returns, trade_markers = run_backtest(
        DummyAgent(),
        bars_df,
        initial_cash,
        strategy,
        "AAPL",
        ["market"],
        {},
        num_workers=1,
    )
    assert len(portfolio_value) == 2
    assert len(daily_returns) == 2
    assert all("date" in d and "value" in d for d in portfolio_value)
    assert isinstance(trade_markers, list)


def test_fetch_bars(monkeypatch):
    # Mock Alpaca data client
    class DummyClient:
        def get_stock_bars(self, bars_request):
            class DummyDF:
                def __init__(self):
                    # Create MultiIndex: (symbol, date)
                    idx = pd.MultiIndex.from_product(
                        [["AAPL"], pd.to_datetime(["2024-01-01", "2024-01-02"])],
                        names=["symbol", "timestamp"],
                    )
                    self.df = pd.DataFrame(
                        {"open": [100, 105], "close": [110, 100]}, index=idx
                    )

                def xs(self, symbol):
                    return self.df.xs(symbol)

                def sort_index(self):
                    return self.df.sort_index()

            return DummyDF()

    bars = fetch_bars(DummyClient(), "AAPL", "2024-01-01", "2024-01-02")
    assert isinstance(bars, pd.DataFrame)
    assert "open" in bars.columns
    assert "close" in bars.columns
