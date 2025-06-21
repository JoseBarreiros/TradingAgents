import pytest
import pandas as pd
from tradingagents.dataflows.utils import save_output, get_current_date, get_next_weekday

def test_save_output(tmp_path):
    """Test that save_output writes a DataFrame to CSV at the given path."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    save_path = tmp_path / "test.csv"
    save_output(df, "test_tag", str(save_path))
    loaded = pd.read_csv(save_path, index_col=0)
    pd.testing.assert_frame_equal(df, loaded)

def test_get_current_date_format():
    """Test that get_current_date returns a string in YYYY-MM-DD format."""
    date_str = get_current_date()
    assert len(date_str) == 10
    assert date_str[4] == "-" and date_str[7] == "-"

def test_get_next_weekday_weekday():
    """Test get_next_weekday returns the same date if it's a weekday."""
    date = "2024-06-20"  # Thursday
    result = get_next_weekday(date)
    assert str(result).startswith("2024-06-20")

def test_get_next_weekday_weekend():
    """Test get_next_weekday returns the next Monday if input is a weekend."""
    date = "2024-06-22"  # Saturday
    result = get_next_weekday(date)
    assert result.weekday() == 0  # Monday