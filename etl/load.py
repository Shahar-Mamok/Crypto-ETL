from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd, yaml

def load_config() -> dict:
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_to_sqlite(df: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    cfg = load_config()
    db_path = cfg["paths"]["db"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)
