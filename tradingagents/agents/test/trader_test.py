import pytest
from unittest.mock import MagicMock
from tradingagents.agents.trader.trader import create_trader

@pytest.fixture
def dummy_state():
    return {
        "company_of_interest": "AAPL",
        "investment_plan": "Buy 10 shares.",
        "market_report": "Market up.",
        "sentiment_report": "Positive.",
        "news_report": "No news.",
        "fundamentals_report": "Strong.",
        "messages": [],
    }

def test_trader_node_returns_investment_plan(dummy_state):
    """
    Test that the trader node returns the expected investment plan string and correct output structure.

    This test sets up MagicMock LLM and memory objects, configures their return values,
    and asserts that the node's output contains the correct 'trader_investment_plan' key and value,
    the correct sender, and that the messages list contains the expected content.
    """    
    llm = MagicMock()
    memory = MagicMock()
    memory.get_memories.return_value = [
        {"recommendation": "Past rec 1"},
        {"recommendation": "Past rec 2"},
    ]
    mock_response = MagicMock()
    mock_response.content = "Trader investment plan"
    llm.invoke.return_value = mock_response

    node = create_trader(llm, memory)
    result = node(dummy_state)
    assert "trader_investment_plan" in result
    assert result["trader_investment_plan"] == "Trader investment plan"
    assert result["sender"] == "Trader"
    assert isinstance(result["messages"], list)
    assert result["messages"][0].content == "Trader investment plan"