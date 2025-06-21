import pytest
from unittest.mock import MagicMock
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher


@pytest.fixture
def dummy_state():
    return {
        "market_report": "Market up.",
        "sentiment_report": "Positive.",
        "news_report": "No news.",
        "fundamentals_report": "Strong.",
        "investment_debate_state": {
            "history": "Previous debate.",
            "bull_history": "Previous bull.",
            "bear_history": "Previous bear.",
            "current_response": "Bear says: risky.",
            "count": 1,
        },
        "messages": [],
    }


def test_bull_researcher_node_updates_investment_debate_state(dummy_state):
    """
    Test that the bull researcher node updates the investment debate state with a bullish argument.

    This test sets up MagicMock LLM and memory objects, configures their return values,
    and asserts that the node's output updates the 'investment_debate_state' with the correct
    history, bull_history, count, and current_response fields reflecting the bull analyst's argument.
    """
    llm = MagicMock()
    memory = MagicMock()
    memory.get_memories.return_value = [
        {"recommendation": "Past rec 1"},
        {"recommendation": "Past rec 2"},
    ]
    llm.invoke.return_value = MagicMock(content="Bullish argument here.")

    node = create_bull_researcher(llm, memory)
    result = node(dummy_state)
    assert "investment_debate_state" in result
    debate = result["investment_debate_state"]
    assert "Bull Analyst: Bullish argument here." in debate["history"]
    assert "Bull Analyst: Bullish argument here." in debate["bull_history"]
    assert debate["count"] == 2
    assert debate["current_response"].startswith("Bull Analyst:")


def test_bear_researcher_node_updates_investment_debate_state(dummy_state):
    """
    Test that the bear researcher node updates the investment debate state with a bearish argument.

    This test sets up MagicMock LLM and memory objects, configures their return values,
    and asserts that the node's output updates the 'investment_debate_state' with the correct
    history, bear_history, count, and current_response fields reflecting the bear analyst's argument.
    """
    llm = MagicMock()
    memory = MagicMock()
    memory.get_memories.return_value = [
        {"recommendation": "Past rec 1"},
        {"recommendation": "Past rec 2"},
    ]
    llm.invoke.return_value = MagicMock(content="Bearish argument here.")

    node = create_bear_researcher(llm, memory)
    result = node(dummy_state)
    assert "investment_debate_state" in result
    debate = result["investment_debate_state"]
    assert "Bear Analyst: Bearish argument here." in debate["history"]
    assert "Bear Analyst: Bearish argument here." in debate["bear_history"]
    assert debate["count"] == 2
    assert debate["current_response"].startswith("Bear Analyst:")
