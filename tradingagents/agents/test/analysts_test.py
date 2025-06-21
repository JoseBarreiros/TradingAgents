import pytest
from unittest.mock import MagicMock

from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.fundamentals_analyst import (
    create_fundamentals_analyst,
)
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.social_media_analyst import (
    create_social_media_analyst,
)
from tradingagents.agents.trader.trader import create_trader


@pytest.fixture
def dummy_state():
    return {
        "trade_date": "2024-06-20",
        "company_of_interest": "AAPL",
        "market_report": "Market up.",
        "sentiment_report": "Positive.",
        "news_report": "No news.",
        "fundamentals_report": "Strong.",
        "investment_plan": "Buy 10 shares.",
        "messages": [],
    }


def test_market_analyst_node_returns_report(dummy_state):
    """
    Test that the market analyst node returns the expected market report string.

    This test sets up a MagicMock LLM and toolkit, configures the toolkit with mock tools,
    and patches the LLM's bind_tools method to return a mock response with a specific content.
    It then asserts that the node's output contains the correct 'market_report' key and value.
    """
    llm = MagicMock()
    toolkit = MagicMock()
    toolkit.config = {"online_tools": True}
    mock_tool = MagicMock()
    mock_tool.name = "get_YFin_data_online"
    mock_tool2 = MagicMock()
    mock_tool2.name = "get_stockstats_indicators_report_online"
    toolkit.get_YFin_data_online = mock_tool
    toolkit.get_stockstats_indicators_report_online = mock_tool2

    # Patch the entire call chain to return the string directly
    mock_response = MagicMock()
    mock_response.content = "Market analysis result"

    def bind_tools_side_effect(*args, **kwargs):
        class Callable:
            def __call__(self, *a, **kw):
                return mock_response

        return Callable()

    llm.bind_tools.side_effect = bind_tools_side_effect

    node = create_market_analyst(llm, toolkit)
    result = node(dummy_state)
    assert "market_report" in result
    assert result["market_report"] == "Market analysis result"


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeChain:
    def invoke(self, *args, **kwargs):
        return FakeResponse("Fundamentals analysis result")


class FakePrompt:
    def partial(self, **kwargs):
        return self

    def __or__(self, other):
        return FakeChain()


class FakeLLM:
    def bind_tools(self, tools):
        def chain_callable(*args, **kwargs):
            return FakeChain()

        return chain_callable


def test_fundamentals_analyst_node_returns_report(dummy_state, monkeypatch):
    """
    Test that the fundamentals analyst node returns the expected fundamentals report string.

    This test patches ChatPromptTemplate.from_messages to return a fake prompt,
    sets up a fake LLM and toolkit, and ensures the node's output contains the correct
    'fundamentals_report' key and value.
    """
    # Patch ChatPromptTemplate.from_messages to return our FakePrompt
    from tradingagents.agents.analysts import fundamentals_analyst

    monkeypatch.setattr(
        fundamentals_analyst.ChatPromptTemplate,
        "from_messages",
        lambda *a, **k: FakePrompt(),
    )

    llm = FakeLLM()

    class FakeTool:
        name = "get_fundamentals_openai"

    toolkit = type("Toolkit", (), {})()
    toolkit.config = {"online_tools": True}
    toolkit.get_fundamentals_openai = FakeTool()

    node = fundamentals_analyst.create_fundamentals_analyst(llm, toolkit)
    result = node(dummy_state)
    assert "fundamentals_report" in result
    assert result["fundamentals_report"] == "Fundamentals analysis result"


def test_news_analyst_node_returns_report(dummy_state, monkeypatch):
    """
    Test that the news analyst node returns the expected news report string.

    This test patches ChatPromptTemplate.from_messages to return a MagicMock prompt,
    configures the prompt and chain mocks to simulate the expected call chain,
    and sets up the toolkit and LLM mocks. It then asserts that the node's output
    contains the correct 'news_report' key and value.
    """
    from unittest.mock import MagicMock

    # Patch ChatPromptTemplate.from_messages to return a MagicMock with __or__ method
    from tradingagents.agents.analysts import news_analyst

    fake_prompt = MagicMock()
    fake_chain = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "News analysis result"
    fake_chain.invoke.return_value = mock_response
    fake_prompt.__or__.return_value = fake_chain
    fake_prompt.partial.return_value = fake_prompt
    monkeypatch.setattr(
        news_analyst.ChatPromptTemplate, "from_messages", lambda *a, **k: fake_prompt
    )

    llm = MagicMock()
    toolkit = MagicMock()
    toolkit.config = {"online_tools": True}
    # Create mock tools with .name attributes
    mock_tool1 = MagicMock()
    mock_tool1.name = "get_global_news_openai"
    mock_tool2 = MagicMock()
    mock_tool2.name = "get_google_news"
    mock_tool3 = MagicMock()
    mock_tool3.name = "get_reddit_news_online"
    mock_tool4 = MagicMock()
    mock_tool4.name = "get_finnhub_news_online"

    toolkit.get_global_news_openai = mock_tool1
    toolkit.get_google_news = mock_tool2
    toolkit.get_reddit_news_online = mock_tool3
    toolkit.get_finnhub_news_online = mock_tool4
    llm.bind_tools.return_value = lambda tools: lambda *a, **k: fake_chain

    node = news_analyst.create_news_analyst(llm, toolkit)
    result = node(dummy_state)
    assert "news_report" in result
    assert result["news_report"] == "News analysis result"


def test_social_media_analyst_node_returns_report(dummy_state, monkeypatch):
    """
    Test that the social media analyst node returns the expected sentiment report string.

    This test patches ChatPromptTemplate.from_messages to return a MagicMock prompt,
    configures the prompt and chain mocks to simulate the expected call chain,
    and sets up the toolkit and LLM mocks. It then asserts that the node's output
    contains the correct 'sentiment_report' key and value.
    """
    from unittest.mock import MagicMock
    from tradingagents.agents.analysts import social_media_analyst

    # Patch ChatPromptTemplate.from_messages to return a MagicMock with __or__ method
    fake_prompt = MagicMock()
    fake_chain = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Social media sentiment result"
    fake_chain.invoke.return_value = mock_response
    fake_prompt.__or__.return_value = fake_chain
    fake_prompt.partial.return_value = fake_prompt
    monkeypatch.setattr(
        social_media_analyst.ChatPromptTemplate,
        "from_messages",
        lambda *a, **k: fake_prompt,
    )

    llm = MagicMock()
    toolkit = MagicMock(
        spec=[
            "config",
            "get_stock_news_openai",
            "get_reddit_news_online",
            "get_twitter_news_online",
        ]
    )
    toolkit.config = {"online_tools": True}

    mock_tool1 = MagicMock()
    mock_tool1.name = "get_stock_news_openai"
    mock_tool2 = MagicMock()
    mock_tool2.name = "get_reddit_news_online"
    mock_tool3 = MagicMock()
    mock_tool3.name = "get_twitter_news_online"
    mock_tool4 = MagicMock()
    mock_tool4.name = "get_reddit_stock_info_online"
    toolkit.get_reddit_stock_info_online = mock_tool4

    toolkit.get_stock_news_openai = mock_tool1
    toolkit.get_reddit_news_online = mock_tool2
    toolkit.get_twitter_news_online = mock_tool3
    toolkit.get_reddit_stock_info_online = mock_tool4

    llm.bind_tools.return_value = lambda tools: lambda *a, **k: fake_chain

    node = social_media_analyst.create_social_media_analyst(llm, toolkit)
    result = node(dummy_state)
    assert "sentiment_report" in result
    assert result["sentiment_report"] == "Social media sentiment result"


def test_trader_node_returns_investment_plan(dummy_state):
    """
    Test that the trader node returns the expected investment plan string.

    This test sets up MagicMock LLM and memory objects, configures their return values,
    and asserts that the node's output contains the correct 'trader_investment_plan' key and value.
    """
    llm = MagicMock()
    memory = MagicMock()
    llm.invoke.return_value = MagicMock(content="Trader investment plan")
    memory.get_memories.return_value = []
    node = create_trader(llm, memory)
    result = node(dummy_state, name="Trader")
    assert "trader_investment_plan" in result
    assert result["trader_investment_plan"] == "Trader investment plan"
