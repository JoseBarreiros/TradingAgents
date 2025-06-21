import pytest
from tradingagents.graph.setup import GraphSetup
from tradingagents.agents.utils.agent_utils import Toolkit
from tradingagents.graph.conditional_logic import ConditionalLogic
from unittest.mock import MagicMock

def test_graph_setup_initializes():
    """
    Test that GraphSetup initializes correctly with the provided arguments,
    and that the risk_level and conditional_logic attributes are set as expected.
    """    
    toolkit = MagicMock(spec=Toolkit)
    logic = ConditionalLogic()
    gs = GraphSetup(
        quick_thinking_llm=MagicMock(),
        deep_thinking_llm=MagicMock(),
        toolkit=toolkit,
        tool_nodes={},
        bull_memory=MagicMock(),
        bear_memory=MagicMock(),
        trader_memory=MagicMock(),
        invest_judge_memory=MagicMock(),
        risk_manager_memory=MagicMock(),
        conditional_logic=logic,
        risk_level="medium",
    )
    assert gs.risk_level == "medium"
    assert gs.conditional_logic is logic