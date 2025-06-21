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
