import pandas as pd

MARKET_COLS = [
    "id","symbol","name","current_price","market_cap","total_volume",
    "price_change_percentage_24h","price_change_percentage_7d_in_currency",
    "last_updated"
]

def transform_markets(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["id","symbol","name","current_price","market_cap","total_volume","pct_change_7d","last_updated"])
    cols = [c for c in MARKET_COLS if c in df.columns]
    df = df[cols].copy()
    if "last_updated" in df:
        df["last_updated"] = pd.to_datetime(df["last_updated"], utc=True, errors="coerce")
    df.rename(columns={"price_change_percentage_7d_in_currency":"pct_change_7d"}, inplace=True)
    return df

def transform_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_ms","price","coin_id","ts"])
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df.dropna(subset=["price","ts"], inplace=True)
    df["coin_id"] = df["coin_id"].astype(str)
    df = df.sort_values("ts")
    return df
