import os
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from alpaca_trade_api.rest import REST

# Setup Alpaca client
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets/"  # os.getenv("APCA_API_BASE_URL")

alpaca = REST(API_KEY, API_SECRET, base_url=BASE_URL)
account = alpaca.get_account()
print(f"Account status: {account.status}")
print(f"Account type: {'LIVE' if 'live' in account.id else 'PAPER'}")

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4.1-nano"  # Use a different model
config["quick_think_llm"] = "gpt-4.1-nano"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds
config["online_tools"] = True  # Increase debate rounds

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
symbol = "NVDA"
date = "2025-06-18"
_, decision = ta.propagate(symbol, date)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns


print(f"Decision for {symbol}: {decision}")

# If agent wants to buy or sell, place paper trade
if decision == "BUY":
    alpaca.submit_order(
        symbol=symbol, qty=1, side="buy", type="market", time_in_force="gtc"
    )
    print(f"Placed BUY order for 1 share of {symbol}")

elif decision == "SELL":
    alpaca.submit_order(
        symbol=symbol, qty=1, side="sell", type="market", time_in_force="gtc"
    )
    print(f"Placed SELL order for 1 share of {symbol}")

else:
    print(f"No action taken: {decision}")
