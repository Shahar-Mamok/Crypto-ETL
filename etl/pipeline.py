
"""High level orchestration for the crypto ETL pipeline.

The original repository shipped with a small script that imported symbols using
implicit relative imports (``from extract import extract``).  Running the module
as a script therefore failed because Python resolves imports relative to the
``PYTHONPATH``.  In addition, the previous implementation had very little
structure which made it difficult to extend the pipeline with additional data
sources.

This refactored module keeps the orchestration extremely small but
intentionally uses package relative imports.  That allows the file to be
executed either via ``python -m etl.pipeline`` or by calling :func:`run_pipeline`
from another module.  The execution flow is unchanged – extract, transform and
load – yet the interfaces are now explicit and therefore easier to reason
about.
"""

from __future__ import annotations

from .extract import extract
from .transform import transform_history, transform_markets
from .load import load_to_sqlite


def run_pipeline() -> None:
    """Execute the end‑to‑end ETL pipeline."""

    # Extract the raw data from all configured sources
    data = extract()

    # Transform into clean tabular structures that the UI can consume
    df_markets = transform_markets(data["markets"])
    df_history = transform_history(data["history"])

    # Load the datasets into the analytics warehouse (SQLite by default)
    load_to_sqlite(df_markets, "markets", if_exists="replace")
    load_to_sqlite(df_history, "history", if_exists="replace")

    print("ETL pipeline completed. Markets and history tables updated.")


if __name__ == "__main__":
    run_pipeline()