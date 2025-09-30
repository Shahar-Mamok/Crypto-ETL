"""Utilities for loading analytics datasets used by the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import yaml


@dataclass(frozen=True)
class DatasetDefinition:
    """Description of an auxiliary dataset that can be surfaced in the UI."""

    id: str
    type: str
    path: Path

    def load(self) -> pd.DataFrame:
        if self.type == "csv":
            return pd.read_csv(self.path)
        raise ValueError(f"Unsupported dataset type: {self.type}")


@dataclass
class DataRepository:
    """Simple data access layer wrapping SQLite and file based sources."""

    db_path: Path
    supplementary_datasets: Iterable[DatasetDefinition]

    def load_table(self, table: str) -> pd.DataFrame:
        import sqlite3

        if not self.db_path.exists():
            return pd.DataFrame()
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(f"SELECT * FROM {table}", conn)

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        for definition in self.supplementary_datasets:
            if definition.id == dataset_id:
                return definition.load()
        raise KeyError(dataset_id)


def load_config(path: str | Path = "configs/config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_repository(cfg: Optional[Dict] = None) -> DataRepository:
    cfg = cfg or load_config()
    paths = cfg.get("paths", {})
    db_path = Path(paths.get("db", "data/warehouse.db"))

    supplementary_defs = [
        DatasetDefinition(
            id=item["id"],
            type=item["type"],
            path=Path(item["path"]),
        )
        for item in cfg.get("supplementary_datasets", [])
    ]

    return DataRepository(db_path=db_path, supplementary_datasets=supplementary_defs)
