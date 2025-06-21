"""
time python backtesting.py --start_date 2025-05-18 --end_date 2025-06-18  --deep_think_llm gpt-4o \
    --quick_think_llm gpt-4.1-mini --initial_cash 10000 \
    --symbol AAPL --num_workers 10 --risk_level medium \
        --selected_analysts market social news fundamentals   

paper experiment:
time python backtesting.py --start_date 2024-01-01 --end_date 2024-03-29  --deep_think_llm gpt-4o \
    --quick_think_llm gpt-4.1-mini --initial_cash 10000 \
    --symbol AAPL --num_workers 1 --risk_level no_guidance \
        --selected_analysts market social news fundamentals  

test:
time python backtesting.py --num_workers 2
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from pathlib import Path
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from copy import deepcopy
from matplotlib.transforms import offset_copy

# Alpaca credentials
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
MODELS = ["gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o"]


def fetch_bars(data_client, symbol, start_date, end_date):
    bars_request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
    )
    bars_df = data_client.get_stock_bars(bars_request).df
    bars_df = bars_df.xs(symbol)
    bars_df = bars_df.sort_index()
    return bars_df


def calculate_metrics(results_df, daily_returns):
    cr = (results_df["value"].iloc[-1] / results_df["value"].iloc[0] - 1) * 100
    num_years = (results_df.index[-1] - results_df.index[0]).days / 365
    arr = (
        (results_df["value"].iloc[-1] / results_df["value"].iloc[0]) ** (1 / num_years)
        - 1
    ) * 100
    daily_return_values = [r["value"] for r in daily_returns]
    mean_return = np.mean(daily_return_values)
    std_return = np.std(daily_return_values)
    sr = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    rolling_max = results_df["value"].cummax()
    drawdown = (results_df["value"] - rolling_max) / rolling_max
    mdd = drawdown.min() * 100
    return cr, arr, sr, mdd


def run_trade_day(
    symbol, trade_date_str, config, selected_analysts, open_price, close_price
):
    # Clone config and inject a unique memory name
    local_config = deepcopy(config)
    # Create a new agent instance inside the thread
    agent = TradingAgentsGraph(
        selected_analysts=selected_analysts, debug=True, config=local_config
    )

    # Get decision
    _, decision = agent.propagate(symbol, trade_date_str)

    result = {
        "date": trade_date_str,
        "decision": decision,
        "open_price": open_price,
        "close_price": close_price,
    }

    return result


def strategy(
    decision,
    trade_date_str,
    open_price,
    close_price,
    cash,
    annual_borrow_rate=0.05,
    trade_commision=0.0025,
):
    # This strategy should be executed daily at the market open, so we only hold cash at night.

    print(f"Decision: {decision} for {trade_date_str}")
    quantity = int(cash // open_price)  # only buy full stocks
    print(f"Calculated stock quantity based on initial cash: {quantity}")
    if decision == "BUY":
        pnl = (close_price - open_price) * quantity
        print(
            f"BUY executed: Bought {quantity} stocks at {open_price}, Sold at {close_price}, PnL = {pnl:.2f}"
        )
    elif decision == "SELL":
        # I first borrow the stock at a premium
        commision = trade_commision * (close_price + open_price)
        premium_for_borrowing_one_stock = (
            open_price * (annual_borrow_rate / 365) + commision
        )
        cost = premium_for_borrowing_one_stock * quantity
        # Inmediately sell
        pnl = (open_price - close_price) * quantity - cost
        print(
            f"SELL executed: Shorted {quantity} stocks at {open_price}, "
            f"Covered at {close_price}, Transaction Cost {cost:.2f}, "
            f"TC per share: {premium_for_borrowing_one_stock:.2f}, PnL = {pnl:.2f}"
        )
    else:
        pnl = 0
        print("HOLD. No trade executed.")
    return pnl


def run_backtest(
    agent,
    bars_df,
    initial_cash,
    strategy,
    symbol,
    selected_analysts,
    config,
    num_workers=1,
):
    cash = initial_cash
    portfolio_value = []
    daily_returns = []
    trade_markers = []

    if num_workers > 1:
        results = parallel_trade_days(
            bars_df, symbol, config, selected_analysts, num_workers
        )
    else:
        results = []
        for trade_date, row in bars_df.iterrows():
            trade_date_str = trade_date.strftime("%Y-%m-%d")
            open_price = row["open"]
            close_price = row["close"]
            _, decision = agent.propagate(symbol, trade_date_str)
            results.append(
                {
                    "date": trade_date_str,
                    "decision": decision,
                    "open_price": open_price,
                    "close_price": close_price,
                }
            )

    results = sorted(results, key=lambda x: x["date"])
    for r in results:
        trade_date_str = r["date"]
        open_price = r["open_price"]
        close_price = r["close_price"]
        decision = r["decision"]
        pnl = strategy(decision, trade_date_str, open_price, close_price, cash)
        if decision in ["BUY", "SELL"]:
            trade_markers.append((trade_date_str, open_price, decision, pnl))
        cash += pnl
        portfolio_value.append({"date": trade_date_str, "value": cash})
        daily_returns.append(
            {
                "date": trade_date_str,
                "value": pnl / (cash - pnl if cash - pnl > 0 else 1),
            }
        )
    return portfolio_value, daily_returns, trade_markers


def parallel_trade_days(bars_df, symbol, config, selected_analysts, num_workers):
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for trade_date, row in bars_df.iterrows():
            trade_date_str = trade_date.strftime("%Y-%m-%d")
            open_price = row["open"]
            close_price = row["close"]
            futures.append(
                executor.submit(
                    run_trade_day,
                    symbol,
                    trade_date_str,
                    config,
                    selected_analysts,
                    open_price,
                    close_price,
                )
            )
        for f in as_completed(futures):
            results.append(f.result())
    return results


def plot_backtest(
    results_df,
    spy_df,
    daily_return_df,
    bars_df,
    trade_markers,
    symbol,
    cr,
    arr,
    sr,
    mdd,
    selected_analysts,
    output_dir,
):
    """
    Plot portfolio value, normalized value, daily return, and asset price with trade markers.

    Saves the plot as 'portfolio_value_plot.png' in output_dir.
    """
    # Plot portfolio value, normalized value, daily return, and asset price
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4,
        1,
        figsize=(12, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
        constrained_layout=True,
    )

    # Portfolio value
    ax1.plot(
        results_df.index, results_df["value"], label="Portfolio Value", linewidth=2
    )
    ax1.scatter(results_df.index, results_df["value"], color="red", s=25)
    for x, y in zip(results_df.index, results_df["value"]):
        ax1.annotate(
            f"${y:.2f}",
            (x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            rotation=0,
            clip_on=False,
        )
    ax1.set_title(
        f"{symbol} Portfolio Value\nCR%: {cr:.2f}, ARR%: {arr:.2f}\nSR: {sr:.2f}, MDD%: {mdd:.2f}\nanalysts: {selected_analysts}"
    )
    ax1.set_ylabel("Portfolio ($)")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Normalized value
    ax2.plot(
        results_df.index,
        results_df["normalized"],
        label="Normalized Portfolio",
        color="green",
        linewidth=2,
    )
    ax2.plot(
        spy_df.index,
        spy_df["normalized"],
        label="SPY Baseline",
        color="gray",
        linestyle="--",
    )
    ax2.scatter(results_df.index, results_df["normalized"], color="darkgreen", s=25)
    for x, y in zip(results_df.index, results_df["normalized"]):
        ax2.annotate(
            f"{y:.4f}",
            (x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            rotation=0,
            clip_on=False,
        )
    ax2.set_ylabel("Normalized")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # Daily returns
    ax3.step(
        daily_return_df.index,
        daily_return_df["value"],
        label="Daily return",
        color="steelblue",
        where="mid",
    )
    ax3.set_ylabel("Daily return")
    ax3.set_xlabel("Date")
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    # Compute lower price for each day (min of open and close)
    bottoms = np.minimum(bars_df["open"], bars_df["close"])
    heights = np.abs(bars_df["close"] - bars_df["open"])
    colors = [
        "green" if close > open_ else "red"
        for open_, close in zip(bars_df["open"], bars_df["close"])
    ]
    # Draw bars from low to high, regardless of direction
    ax4.bar(
        bars_df.index,
        heights,
        bottom=bottoms,
        color=colors,
        width=0.6,
        align="center",
        alpha=0.6,
        label="Open to Close Spread",
    )
    ax4.plot(
        bars_df.index,
        bars_df["open"],
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label="Open Price",
    )
    ax4.set_ylabel(f"{symbol} Price ($)")
    ax4.set_title(f"{symbol} Daily Open → Close Spread")
    ax4.grid(True, linestyle="--", alpha=0.6)

    # Add BUY/SELL markers at open price
    for date_str, open_price, decision, pnl in trade_markers:
        date_match = bars_df.index[bars_df.index.strftime("%Y-%m-%d") == date_str]
        if len(date_match) == 0:
            continue
        date = date_match[0]
        color = "green" if decision == "BUY" else "red"
        marker = "^" if decision == "BUY" else "v"
        y_offset = 12
        ax4.plot(
            date,
            open_price,
            marker=marker,
            color=color,
            markersize=10,
            label=decision,
            zorder=5,
        )
        ax4.annotate(
            f"pnl:{pnl:.2f}",
            (date, open_price),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=color,
            zorder=6,
            weight="bold",
        )

    # Set y-axis limits with 20% padding
    min_price = min(bars_df[["open", "close"]].min())
    max_price = max(bars_df[["open", "close"]].max())
    padding = 0.20 * (max_price - min_price)
    ax4.set_ylim(min_price - padding, max_price + padding)

    # Prevent duplicate legend entries
    handles, labels = ax4.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax4.legend(unique.values(), unique.keys())

    # Format x-axis dates
    ax4.xaxis.set_major_locator(mdates.DayLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=30)

    # Save figure
    plot_path = output_dir / "portfolio_value_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved as {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a TradingAgents backtest using Alpaca data."
    )
    parser.add_argument(
        "--start_date", default="2025-06-15", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--symbol", default="NVDA", help="stock ticket")
    parser.add_argument(
        "--end_date", default="2025-06-18", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output_path", default="./results", help="Output path for the plot image"
    )
    parser.add_argument("--deep_think_llm", default="gpt-4.1-nano", choices=MODELS)
    parser.add_argument("--quick_think_llm", default="gpt-4.1-nano", choices=MODELS)
    parser.add_argument("--initial_cash", default=10000.0, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--selected_analysts",
        nargs="+",
        default=["market"],
        choices=["market", "social", "news", "fundamentals"],
        help="List of analysts to use.",
    )
    parser.add_argument(
        "--risk_level",
        default="medium",
        choices=["low", "medium", "high", "no_guidance"],
        help="Risk level for the trading strategy (low, medium, high, no_guidance)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_path) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Alpaca data client
    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

    # Configure TradingAgentsGraph
    config = DEFAULT_CONFIG.copy()
    config["deep_think_llm"] = args.deep_think_llm
    config["quick_think_llm"] = args.quick_think_llm
    config["max_debate_rounds"] = 1
    config["online_tools"] = True
    config["risk_level"] = args.risk_level

    agent = TradingAgentsGraph(
        selected_analysts=args.selected_analysts, debug=True, config=config
    )

    # Backtest parameters
    symbol = args.symbol
    start_date = args.start_date
    end_date = args.end_date
    initial_cash = args.initial_cash

    # Request historical bars
    print("Fetching historical bars...")
    bars_df = fetch_bars(data_client, symbol, start_date, end_date)
    print(f"Retrieved {len(bars_df)} trading days of data.")

    # Request SPY as baseline
    spy_request = StockBarsRequest(
        symbol_or_symbols="SPY", timeframe=TimeFrame.Day, start=start_date, end=end_date
    )
    spy_df = data_client.get_stock_bars(spy_request).df
    spy_df = spy_df.xs("SPY")
    spy_df = spy_df.sort_index()
    spy_df["normalized"] = spy_df["close"] / spy_df["close"].iloc[0]

    # Backtest loop
    print("\n--- Backtesting ---")
    portfolio_value, daily_returns, trade_markers = run_backtest(
        agent,
        bars_df,
        initial_cash,
        strategy,
        symbol,
        args.selected_analysts,
        config,
        args.num_workers,
    )

    # Convert to DataFrame
    results_df = pd.DataFrame(portfolio_value)
    results_df["normalized"] = results_df["value"] / initial_cash
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df.set_index("date", inplace=True)

    daily_return_df = pd.DataFrame(daily_returns)
    daily_return_df["date"] = pd.to_datetime(daily_return_df["date"])
    daily_return_df.set_index("date", inplace=True)

    # Calculate metrics
    cr, arr, sr, mdd = calculate_metrics(results_df, daily_returns)

    # Output metrics
    print("\n--- Backtest Performance Metrics ---")
    print(f"Cumulative Return (CR%): {cr:.2f}%")
    print(f"Annualized Return (ARR%): {arr:.2f}%")
    print(f"Sharpe Ratio (SR): {sr:.2f}")
    print(f"Max Drawdown (MDD%): {mdd:.2f}%")

    # Save metrics to text file
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("--- Backtest Performance Metrics ---\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Start Date: {start_date}\n")
        f.write(f"End Date: {end_date}\n")
        f.write(f"Cumulative Return (CR%): {cr:.2f}%\n")
        f.write(f"Annualized Return (ARR%): {arr:.2f}%\n")
        f.write(f"Sharpe Ratio (SR): {sr:.2f}\n")
        f.write(f"Max Drawdown (MDD%): {mdd:.2f}%\n")
    print(f"Metrics saved to {metrics_path}")

    # Plot portfolio value, normalized value, daily return, and asset price
    plot_backtest(
        results_df,
        spy_df,
        daily_return_df,
        bars_df,
        trade_markers,
        symbol,
        cr,
        arr,
        sr,
        mdd,
        args.selected_analysts,
        output_dir,
    )


if __name__ == "__main__":
    main()
