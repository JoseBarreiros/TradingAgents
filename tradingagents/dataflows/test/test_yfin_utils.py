import pytest
from unittest.mock import patch, MagicMock
from tradingagents.dataflows.yfin_utils import YFinanceUtils

@patch("tradingagents.dataflows.yfin_utils.yf")
def test_get_stock_data_calls_history(mock_yf):
    """Test get_stock_data calls yfinance.Ticker.history with correct params."""
    mock_ticker = MagicMock()
    mock_yf.Ticker.return_value = mock_ticker
    mock_ticker.history.return_value = "data"
    utils = YFinanceUtils()
    result = utils.get_stock_data("AAPL", "2024-01-01", "2024-01-10")
    assert result == "data"
    mock_ticker.history.assert_called()

@patch("tradingagents.dataflows.yfin_utils.yf")
def test_get_stock_info_returns_dict(mock_yf):
    """Test get_stock_info returns a dictionary."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"sector": "Tech"}
    mock_yf.Ticker.return_value = mock_ticker
    result = YFinanceUtils.get_stock_info("AAPL")
    assert isinstance(result, dict)
    assert result["sector"] == "Tech"