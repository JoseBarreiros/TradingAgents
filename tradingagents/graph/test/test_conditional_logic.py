import pytest
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.agents.utils.agent_states import AgentState


@pytest.fixture
def dummy_state():
    """
    Fixture that returns a sample state dictionary mimicking the structure
    expected by ConditionalLogic methods for risk and investment debate.
    """
    return {
        "risk_debate_state": {
            "count": 3,
            "latest_speaker": "Risky",
        },
        "investment_debate_state": {
            "count": 2,
            "current_response": "Bull",
        },
        "messages": [{"tool_calls": False}],
    }


def test_should_continue_risk_analysis_to_risk_judge(dummy_state):
    """
    Test that should_continue_risk_analysis returns 'Risk Judge'
    when the risk debate count exceeds the max_risk_discuss_rounds.
    """
    logic = ConditionalLogic(max_risk_discuss_rounds=1)
    dummy_state["risk_debate_state"]["count"] = 3
    assert logic.should_continue_risk_analysis(dummy_state) == "Risk Judge"


def test_should_continue_risk_analysis_to_safe_analyst(dummy_state):
    """
    Test that should_continue_risk_analysis returns 'Safe Analyst'
    when the latest speaker is 'Risky' and the debate should continue.
    """
    logic = ConditionalLogic(max_risk_discuss_rounds=2)
    dummy_state["risk_debate_state"]["count"] = 1
    dummy_state["risk_debate_state"]["latest_speaker"] = "Risky"
    assert logic.should_continue_risk_analysis(dummy_state) == "Safe Analyst"


def test_should_continue_debate_to_bear_researcher(dummy_state):
    """
    Test that should_continue_debate returns 'Bear Researcher'
    when the investment debate count equals max_debate_rounds
    and the current response is 'Bull'.
    """
    logic = ConditionalLogic(max_debate_rounds=1)
    dummy_state["investment_debate_state"]["count"] = 1
    dummy_state["investment_debate_state"]["current_response"] = "Bull"
    assert logic.should_continue_debate(dummy_state) == "Bear Researcher"
