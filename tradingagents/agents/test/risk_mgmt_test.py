import pytest
from unittest.mock import MagicMock
from tradingagents.agents.risk_mgmt.aggresive_debator import create_risky_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_safe_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator

@pytest.fixture
def dummy_state():
    return {
        "market_report": "Market up.",
        "sentiment_report": "Positive.",
        "news_report": "No news.",
        "fundamentals_report": "Strong.",
        "trader_investment_plan": "Buy 10 shares.",
        "risk_debate_state": {
            "history": "Previous debate.",
            "risky_history": "Previous risky.",
            "safe_history": "Previous safe.",
            "neutral_history": "Previous neutral.",
            "latest_speaker": "",
            "current_risky_response": "",
            "current_safe_response": "",
            "current_neutral_response": "",
            "count": 1,
        },
        "messages": [],
    }

def test_risky_debator_node_updates_risk_debate_state(dummy_state):
    """
    Test that the risky debator node updates the risk debate state with a risky argument.

    This test sets up a MagicMock LLM, configures its return value, and asserts that the node's output
    updates the 'risk_debate_state' with the correct history, risky_history, latest_speaker, count,
    and current_risky_response fields reflecting the risky analyst's argument.
    """    
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Risky argument here.")
    node = create_risky_debator(llm)
    result = node(dummy_state)
    assert "risk_debate_state" in result
    debate = result["risk_debate_state"]
    assert "Risky Analyst: Risky argument here." in debate["history"]
    assert "Risky Analyst: Risky argument here." in debate["risky_history"]
    assert debate["latest_speaker"] == "Risky"
    assert debate["count"] == 2
    assert debate["current_risky_response"].startswith("Risky Analyst:")

def test_safe_debator_node_updates_risk_debate_state(dummy_state):
    """
    Test that the safe debator node updates the risk debate state with a safe argument.

    This test sets up a MagicMock LLM, configures its return value, and asserts that the node's output
    updates the 'risk_debate_state' with the correct history, safe_history, latest_speaker, count,
    and current_safe_response fields reflecting the safe analyst's argument.
    """    
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Safe argument here.")
    node = create_safe_debator(llm)
    result = node(dummy_state)
    assert "risk_debate_state" in result
    debate = result["risk_debate_state"]
    assert "Safe Analyst: Safe argument here." in debate["history"]
    assert "Safe Analyst: Safe argument here." in debate["safe_history"]
    assert debate["latest_speaker"] == "Safe"
    assert debate["count"] == 2
    assert debate["current_safe_response"].startswith("Safe Analyst:")

def test_neutral_debator_node_updates_risk_debate_state(dummy_state):
    """
    Test that the neutral debator node updates the risk debate state with a neutral argument.

    This test sets up a MagicMock LLM, configures its return value, and asserts that the node's output
    updates the 'risk_debate_state' with the correct history, neutral_history, latest_speaker, count,
    and current_neutral_response fields reflecting the neutral analyst's argument.
    """    
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Neutral argument here.")
    node = create_neutral_debator(llm)
    result = node(dummy_state)
    assert "risk_debate_state" in result
    debate = result["risk_debate_state"]
    assert "Neutral Analyst: Neutral argument here." in debate["history"]
    assert "Neutral Analyst: Neutral argument here." in debate["neutral_history"]
    assert debate["latest_speaker"] == "Neutral"
    assert debate["count"] == 2
    assert debate["current_neutral_response"].startswith("Neutral Analyst:")