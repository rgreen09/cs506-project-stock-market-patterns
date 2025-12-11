from datetime import time as dt_time

import pandas as pd


def is_within_trading_hours(
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
    market_open: str = "09:30",
    market_close: str = "16:00",
) -> bool:
    """
    Check if a window is fully within trading hours on the same day.
    """
    start_date = start_timestamp.date()
    end_date = end_timestamp.date()
    if start_date != end_date:
        return False

    open_hour, open_minute = map(int, market_open.split(":"))
    close_hour, close_minute = map(int, market_close.split(":"))
    market_open_time = dt_time(open_hour, open_minute)
    market_close_time = dt_time(close_hour, close_minute)

    start_time = start_timestamp.time()
    end_time = end_timestamp.time()
    return (
        market_open_time <= start_time <= market_close_time
        and market_open_time <= end_time <= market_close_time
    )


def spans_multiple_days(start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp) -> bool:
    return start_timestamp.date() != end_timestamp.date()

