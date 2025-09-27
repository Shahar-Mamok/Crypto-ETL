from __future__ import annotations
import os
import json
import time
from pathlib import Path
from typing import Dict, List
import requests
import pandas as pd
import yaml

def load_config() -> dict:
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def fetch_markets(vs_currency: str, per_page: int = 50) -> List[dict]:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    r = requests.get(url, params={
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "price_change_percentage": "1h,24h,7d"
    }, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_coin_history(coin_id: str, vs_currency: str, days: int) -> Dict:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    r = requests.get(url, params={"vs_currency": vs_currency, "days": days}, timeout=30)
    r.raise_for_status()
    return r.json()

def extract() -> Dict[str, pd.DataFrame]:
    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]; ensure_dir(raw_dir)
    vs = cfg["coingecko"]["vs_currency"]
    coin_ids = cfg["coingecko"]["coin_ids"]; days = cfg["coingecko"]["days"]

    # Markets snapshot
    markets = fetch_markets(vs_currency=vs, per_page=50)
    Path(raw_dir, f"markets_{int(time.time())}.json").write_text(
        json.dumps(markets, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    df_markets = pd.json_normalize(markets)

    # History for selected coins
    hist_frames = []
    for cid in coin_ids:
        hist = fetch_coin_history(cid, vs_currency=vs, days=days)
        Path(raw_dir, f"{cid}_hist_{int(time.time())}.json").write_text(
            json.dumps(hist)[:2_000_000], encoding="utf-8"
        )
        prices = hist.get("prices", [])  # [[ts_ms, price], ...]
        df = pd.DataFrame(prices, columns=["ts_ms", "price"])
        df["coin_id"] = cid
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        hist_frames.append(df)
        time.sleep(0.2)

    df_history = pd.concat(hist_frames, ignore_index=True)
    return {
        "markets": df_markets,
        "history": df_history
    }

if __name__ == "__main__":
    out = extract()
    print({k: v.shape for k, v in out.items()})
