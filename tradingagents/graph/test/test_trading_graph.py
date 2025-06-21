import pytest
from unittest.mock import MagicMock, patch
from tradingagents.graph.trading_graph import TradingAgentsGraph

def _full_final_state():
    return {
        "company_of_interest": "AAPL",
        "trade_date": "2024-01-01",
        "market_report": "Market is bullish.",
        "sentiment_report": "Positive sentiment.",
        "news_report": "No major news.",
        "fundamentals_report": "Strong fundamentals.",
        "investment_debate_state": {
            "bull_history": [],
            "bear_history": [],
            "history": [],
            "current_response": "Bullish",
            "judge_decision": "Buy",
        },
        "trader_investment_plan": "Buy 10 shares.",
        "risk_debate_state": {
            "risky_history": [],
            "safe_history": [],
            "neutral_history": [],
            "history": [],
            "judge_decision": "Buy",
        },
        "investment_plan": "Buy 10 shares.",
        "final_trade_decision": "BUY",
    }

def test_trading_agents_graph_initialization():
    """
    Test that TradingAgentsGraph initializes with the expected attributes:
    'graph', 'config', and 'toolkit'.
    """    
    graph = TradingAgentsGraph()
    assert hasattr(graph, "graph")
    assert hasattr(graph, "config")
    assert hasattr(graph, "toolkit")

@patch("tradingagents.graph.trading_graph.Propagator")
def test_propagate_runs_and_returns_state(mock_propagator):
    """
    Test that the propagate method runs and returns a result containing
    'final_trade_decision' when the graph's invoke method is mocked.
    """
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = _full_final_state()
    mock_propagator.return_value.create_initial_state.return_value = {}
    mock_propagator.return_value.get_graph_args.return_value = {}
    with patch("tradingagents.graph.trading_graph.GraphSetup") as mock_setup:
        mock_setup.return_value.setup_graph.return_value = mock_graph
        tg = TradingAgentsGraph()
        tg.graph = mock_graph
        result, signal = tg.propagate("AAPL", "2024-01-01")
        assert "final_trade_decision" in result
