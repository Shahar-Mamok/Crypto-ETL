import pandas as pd

def validate_markets(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise AssertionError("markets: dataframe is empty")
    assert not df["id"].isna().any(), "markets: missing id"
    assert (df["current_price"] >= 0).all(), "markets: negative current_price"

def validate_history(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise AssertionError("history: dataframe is empty")
    assert not df["coin_id"].isna().any(), "history: missing coin_id"
    assert (df["price"] >= 0).all(), "history: negative price"
