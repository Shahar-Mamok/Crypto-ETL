import pandas as pd


# Include all relevant columns for UI profile (including image, ath, atl, supply, etc.)
MARKET_COLS = [
    "id", "symbol", "name", "image", "current_price", "market_cap", "market_cap_rank",
    "fully_diluted_valuation", "total_volume", "high_24h", "low_24h", "price_change_24h",
    "price_change_percentage_24h", "market_cap_change_24h", "market_cap_change_percentage_24h",
    "circulating_supply", "total_supply", "max_supply", "ath", "ath_change_percentage", "ath_date",
    "atl", "atl_change_percentage", "atl_date", "roi", "last_updated", "price_change_percentage_1h_in_currency",
    "price_change_percentage_24h_in_currency", "price_change_percentage_7d_in_currency"
]

def transform_markets(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=MARKET_COLS)
    cols = [c for c in MARKET_COLS if c in df.columns]
    df = df[cols].copy()
    if "last_updated" in df:
        df["last_updated"] = pd.to_datetime(df["last_updated"], utc=True, errors="coerce")
    # Optionally: rename columns for UI if needed
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
