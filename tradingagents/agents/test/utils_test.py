import pytest
from unittest.mock import MagicMock, patch
from tradingagents.agents.utils.agent_utils import Toolkit, create_msg_delete


def test_toolkit_config_update_and_access():
    """
    Test that Toolkit.update_config correctly updates the configuration and that
    a Toolkit instance reflects the updated config values.

    This test sets a custom config, creates a Toolkit instance, and asserts that
    the config values are as expected.
    """
    custom_config = {"online_tools": False, "foo": "bar"}
    Toolkit.update_config(custom_config)
    toolkit = Toolkit()
    assert toolkit.config["online_tools"] is False
    assert toolkit.config["foo"] == "bar"


def test_create_msg_delete_removes_all_messages():
    """
    Test that create_msg_delete returns a function which, when called with a state,
    returns a new state containing all original messages (by id).

    This test simulates a state with messages having .id attributes, applies the delete function,
    and asserts that the returned state's messages list contains the correct ids.
    """

    # Simulate state with messages having .id attributes
    class DummyMsg:
        def __init__(self, id):
            self.id = id

    state = {"messages": [DummyMsg("a"), DummyMsg("b")]}
    delete_fn = create_msg_delete()
    result = delete_fn(state)
    assert "messages" in result
    assert all(hasattr(m, "id") for m in result["messages"])
    assert [m.id for m in result["messages"]] == ["a", "b"]


@patch("tradingagents.agents.utils.agent_utils.interface.get_reddit_global_news")
def test_toolkit_get_reddit_news_calls_interface(mock_get_news):
    """
    Test that Toolkit.get_reddit_news calls the interface.get_reddit_global_news function
    and returns its result.

    This test patches the interface function, sets its return value, calls the Toolkit method,
    and asserts that the result matches and the interface function was called once.
    """
    mock_get_news.return_value = "Some news"
    result = Toolkit.get_reddit_news("2024-06-20")
    assert result == "Some news"
    mock_get_news.assert_called_once()


@patch("tradingagents.agents.utils.agent_utils.interface.get_finnhub_news")
def test_toolkit_get_finnhub_news_calls_interface(mock_get_news):
    """
    Test that Toolkit.get_finnhub_news calls the interface.get_finnhub_news function
    and returns its result.

    This test patches the interface function, sets its return value, calls the Toolkit tool's
    invoke method with the correct arguments, and asserts that the result matches and the
    interface function was called once.
    """
    mock_get_news.return_value = "Finnhub news"
    result = Toolkit.get_finnhub_news.invoke(
        {"ticker": "AAPL", "start_date": "2024-06-01", "end_date": "2024-06-20"}
    )
    assert result == "Finnhub news"
    mock_get_news.assert_called_once()
