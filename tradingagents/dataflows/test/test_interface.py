import pytest
from unittest.mock import patch, MagicMock
from tradingagents.dataflows import interface


@patch("tradingagents.dataflows.interface.fetch_top_from_category")
def test_get_reddit_global_news_empty(mock_fetch):
    """
    Test get_reddit_global_news returns an empty string if no posts are found.
    """
    mock_fetch.return_value = []
    result = interface.get_reddit_global_news("2024-06-20", 1, 5)
    assert result == ""


@patch("tradingagents.dataflows.interface.fetch_top_from_category")
def test_get_reddit_global_news_with_posts(mock_fetch):
    """
    Test get_reddit_global_news returns a formatted string if posts exist.
    """
    mock_fetch.return_value = [
        {"title": "Headline 1", "content": "Some news content."},
        {"title": "Headline 2", "content": ""},
    ]
    result = interface.get_reddit_global_news("2024-06-20", 1, 5)
    assert isinstance(result, str)
    assert "Headline 1" in result
    assert "Some news content." in result
    assert "Headline 2" in result


def test_get_finnhub_news_online():
    result = interface.get_finnhub_news_online("AAPL", "2025-06-20", 5)
    print(result)
    assert (
        isinstance(result, str) and result.strip() != ""
    ), "Result must be a non-empty"


def test_get_google_news():
    result = interface.get_google_news("AAPL", "2025-06-20", 5)
    print(result)
    assert (
        isinstance(result, str) and result.strip() != ""
    ), "Result must be a non-empty"


def test_get_reddit_global_news_online():
    result = interface.get_reddit_global_news("2025-06-20", 1, 5, True)
    print(result)
    assert (
        isinstance(result, str) and result.strip() != ""
    ), "Result must be a non-empty"


def test_get_reddit_company_news_online():
    result = interface.get_reddit_company_news("ASTS", "2025-06-20", 1, 5, True)
    print(result)
    assert (
        isinstance(result, str) and result.strip() != ""
    ), "Result must be a non-empty"


def test_get_stock_stats_indicators_window():
    # online mode
    result = interface.get_stock_stats_indicators_window(
        "AAPL", "close_50_sma", "2025-01-01", 10, True
    )
    print(result)
    assert (
        isinstance(result, str) and result.strip() != ""
    ), "Result must be a non-empty"


def test_get_stockstats_indicator():
    # online mode
    result = interface.get_stockstats_indicator(
        "AAPL", "close_50_sma", "2025-01-01", True
    )
    print(result)
    assert (
        isinstance(result, str) and result.strip() != ""
    ), "Result must be a non-empty string"


def test_get_YFin_data_online():
    # online mode
    result = interface.get_YFin_data_online("AAPL", "2025-01-01", "2025-01-10")
    print(result)
    assert (
        isinstance(result, str) and result.strip() != ""
    ), "Result must be a non-empty string"


def test_get_stock_news_openai():
    # online mode
    result = interface.get_stock_news_openai("AAPL", "2025-01-01")
    print(result)
    assert (
        isinstance(result, str) and result.strip() != ""
    ), "Result must be a non-empty string"


def test_get_global_news_openai():
    # online mode
    result = interface.get_global_news_openai("2025-01-01")
    print(result)
    assert (
        isinstance(result, str) and result.strip() != ""
    ), "Result must be a non-empty string"


def test_get_fundamentals_openai():
    # online mode
    result = interface.get_fundamentals_openai("AAPL", "2025-01-01")
    print(result)
    assert (
        isinstance(result, str) and result.strip() != ""
    ), "Result must be a non-empty string"
