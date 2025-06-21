import pytest
from unittest.mock import MagicMock
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.managers.risk_manager import create_risk_manager


@pytest.fixture
def dummy_state():
    return {
        "company_of_interest": "AAPL",
        "trade_date": "2024-06-20",
        "market_report": "Market up.",
        "sentiment_report": "Positive.",
        "news_report": "No news.",
        "fundamentals_report": "Strong.",
        "investment_debate_state": {
            "history": "",
            "bear_history": "",
            "bull_history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 1,
        },
        "risk_debate_state": {
            "risky_history": "",
            "safe_history": "",
            "neutral_history": "",
            "history": "",
            "latest_speaker": "",
            "current_risky_response": "",
            "current_safe_response": "",
            "current_neutral_response": "",
            "judge_decision": "",
            "count": 1,
        },
        "investment_plan": "Buy 10 shares.",
        "messages": [],
    }


def test_research_manager_node_returns_investment_plan(dummy_state):
    """
    Test that the research manager node returns the expected investment plan string.

    This test sets up MagicMock LLM and memory objects, configures their return values,
    and asserts that the node's output contains the correct 'investment_plan' key and value,
    and that the 'investment_debate_state' is updated with the judge's decision.
    """
    llm = MagicMock()
    memory = MagicMock()
    llm.invoke.return_value = MagicMock(content="Research manager plan")
    memory.get_memories.return_value = [{"recommendation": "Past rec"}]
    node = create_research_manager(llm, memory)
    result = node(dummy_state)
    assert "investment_plan" in result
    assert result["investment_plan"] == "Research manager plan"
    assert "investment_debate_state" in result
    assert (
        result["investment_debate_state"]["judge_decision"] == "Research manager plan"
    )


def test_risk_manager_node_returns_final_trade_decision(dummy_state):
    """
    Test that the risk manager node returns the expected final trade decision string.

    This test sets up MagicMock LLM and memory objects, configures their return values,
    and asserts that the node's output contains the correct 'final_trade_decision' key and value,
    and that the 'risk_debate_state' is updated with the judge's decision.
    """
    llm = MagicMock()
    memory = MagicMock()
    llm.invoke.return_value = MagicMock(content="BUY")
    memory.get_memories.return_value = [{"recommendation": "Past rec"}]
    node = create_risk_manager(llm, memory, risk_level="medium")
    result = node(dummy_state)
    assert "final_trade_decision" in result
    assert result["final_trade_decision"] == "BUY"
    assert "risk_debate_state" in result
    assert result["risk_debate_state"]["judge_decision"] == "BUY"
