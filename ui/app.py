from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from data_access import build_repository, load_config


# -----------------------------------------------------------------------------
# Page configuration & high level metadata
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Intelligence Workbench",
    page_icon="💹",
    layout="wide",
)

st.title("💹 Crypto Intelligence Workbench")
st.caption(
    "A modular analytics surface blending ETL data engineering with a fully configurable dashboard."
)


# -----------------------------------------------------------------------------
# Data access helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_config_cached() -> Dict[str, Any]:
    return load_config()


def _ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "dashboard"


config = _load_config_cached()
repository = build_repository(config)
dashboards_dir = _ensure_directory(config.get("paths", {}).get("dashboards_dir", "data/dashboards"))


def _load_table_cached(db_path: str, table: str) -> pd.DataFrame:
    @st.cache_data(ttl=300, show_spinner=False)
    def _load(path: str, tbl: str) -> pd.DataFrame:
        import sqlite3

        db = Path(path)
        if not db.exists():
            return pd.DataFrame()
        with sqlite3.connect(db) as conn:
            return pd.read_sql(f"SELECT * FROM {tbl}", conn)

    return _load(db_path, table)


def _load_csv_cached(path: str) -> pd.DataFrame:
    @st.cache_data(show_spinner=False)
    def _load(p: str) -> pd.DataFrame:
        file_path = Path(p)
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_csv(file_path)

    return _load(path)


# -----------------------------------------------------------------------------
# Dashboard domain models
# -----------------------------------------------------------------------------
@dataclass
class DashboardContext:
    markets: pd.DataFrame
    history: pd.DataFrame
    portfolio: pd.DataFrame
    filters: Dict[str, Any]


@dataclass
class WidgetDefinition:
    id: str
    label: str
    description: str
    default_config: Dict[str, Any]
    configure: Callable[[DashboardContext, Dict[str, Any]], Dict[str, Any]]
    render: Callable[[DashboardContext, Dict[str, Any]], None]


def _format_currency(value: float) -> str:
    return f"${value:,.0f}" if pd.notnull(value) else "-"


def _format_percentage(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%" if pd.notnull(value) else "-"


METRIC_CATALOG: Dict[str, Dict[str, Any]] = {
    "total_market_cap": {
        "label": "Total Market Cap",
        "compute": lambda df: df.get("market_cap", pd.Series(dtype=float)).sum(),
        "formatter": _format_currency,
    },
    "avg_price": {
        "label": "Average Price",
        "compute": lambda df: df.get("current_price", pd.Series(dtype=float)).mean(),
        "formatter": lambda v: f"${v:,.2f}" if pd.notnull(v) else "-",
    },
    "volume_24h": {
        "label": "24h Volume",
        "compute": lambda df: df.get("total_volume", pd.Series(dtype=float)).sum(),
        "formatter": _format_currency,
    },
    "avg_change_24h": {
        "label": "Avg 24h Change",
        "compute": lambda df: df.get("price_change_percentage_24h", pd.Series(dtype=float)).mean(),
        "formatter": lambda v: _format_percentage(v, digits=2),
    },
    "avg_change_7d": {
        "label": "Avg 7d Change",
        "compute": lambda df: df.get("price_change_percentage_7d_in_currency", pd.Series(dtype=float)).mean(),
        "formatter": lambda v: _format_percentage(v, digits=2),
    },
}


# -----------------------------------------------------------------------------
# Widget configuration helpers
# -----------------------------------------------------------------------------
def _configure_market_kpis(context: DashboardContext, config: Dict[str, Any]) -> Dict[str, Any]:
    metrics = st.multiselect(
        "Metrics",
        options=list(METRIC_CATALOG.keys()),
        format_func=lambda key: METRIC_CATALOG[key]["label"],
        default=config.get("metrics") or ["total_market_cap", "volume_24h", "avg_change_24h"],
        key=f"metrics_{id(context)}",
    )
    return {**config, "metrics": metrics}


def _render_market_kpis(context: DashboardContext, config: Dict[str, Any]) -> None:
    df = context.markets
    metrics: List[str] = config.get("metrics") or []
    if df.empty or not metrics:
        st.info("No market data available for KPI computation.")
        return

    cols = st.columns(len(metrics)) if len(metrics) > 1 else [st]
    for idx, metric_id in enumerate(metrics):
        metric_def = METRIC_CATALOG[metric_id]
        value = metric_def["compute"](df)
        formatted = metric_def["formatter"](value)
        label = metric_def["label"]
        target_col = cols[idx % len(cols)]
        with target_col:
            st.metric(label=label, value=formatted)


def _configure_price_chart(context: DashboardContext, config: Dict[str, Any]) -> Dict[str, Any]:
    available_coins = context.filters.get("available_coins", [])
    default_coins = config.get("coins") or context.filters.get("selected_coins", available_coins[:3])
    coins = st.multiselect(
        "Coins",
        options=available_coins,
        default=default_coins,
        key=f"chart_coins_{id(context)}",
    )
    chart_mode = st.selectbox(
        "Chart type",
        options=["line", "area"],
        index=0 if config.get("chart_type", "line") == "line" else 1,
        key=f"chart_type_{id(context)}",
    )
    normalize = st.checkbox(
        "Normalize prices (rebased to 100)",
        value=config.get("normalize", False),
        key=f"chart_norm_{id(context)}",
    )
    return {**config, "coins": coins, "chart_type": chart_mode, "normalize": normalize}


def _render_price_chart(context: DashboardContext, config: Dict[str, Any]) -> None:
    df = context.history
    if df.empty:
        st.info("No historical price data available.")
        return

    coins = config.get("coins") or context.filters.get("selected_coins")
    if not coins:
        st.info("Select at least one coin to render the price chart.")
        return

    df_plot = df[df["coin_id"].isin(coins)].copy()
    if df_plot.empty:
        st.warning("No data for the selected coins within the chosen date range.")
        return

    if config.get("normalize"):
        df_plot = df_plot.sort_values(["coin_id", "ts"])
        df_plot["price_norm"] = (
            df_plot.groupby("coin_id")["price"].transform(lambda s: s / s.iloc[0] * 100)
        )
        y_col = "price_norm"
        y_title = "Price (rebased to 100)"
    else:
        y_col = "price"
        y_title = "Price (USD)"

    chart_type = config.get("chart_type", "line")
    if chart_type == "area":
        fig = px.area(df_plot, x="ts", y=y_col, color="coin_id", template="plotly_white")
    else:
        fig = px.line(df_plot, x="ts", y=y_col, color="coin_id", template="plotly_white")

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=420,
        legend_title="Coin",
        yaxis_title=y_title,
        xaxis_title="Date",
    )
    st.plotly_chart(fig, use_container_width=True)


def _configure_market_table(context: DashboardContext, config: Dict[str, Any]) -> Dict[str, Any]:
    df = context.markets
    if df.empty:
        return config

    available_cols = [c for c in df.columns if c not in {"image"}]
    default_cols = config.get("columns") or [
        "name",
        "symbol",
        "current_price",
        "market_cap",
        "total_volume",
        "price_change_percentage_24h",
    ]
    columns = st.multiselect(
        "Columns",
        options=available_cols,
        default=[c for c in default_cols if c in available_cols],
        key=f"table_cols_{id(context)}",
    )
    top_n = st.slider(
        "Rows to display",
        min_value=5,
        max_value=min(50, len(df)),
        value=min(config.get("top_n", 10), len(df)),
        step=5,
        key=f"table_topn_{id(context)}",
    )
    sort_column = st.selectbox(
        "Sort by",
        options=columns or available_cols,
        index=0,
        key=f"table_sort_{id(context)}",
    )
    descending = st.checkbox(
        "Sort descending",
        value=config.get("descending", True),
        key=f"table_desc_{id(context)}",
    )
    return {
        **config,
        "columns": columns,
        "top_n": top_n,
        "sort_column": sort_column,
        "descending": descending,
    }


def _render_market_table(context: DashboardContext, config: Dict[str, Any]) -> None:
    df = context.markets
    if df.empty:
        st.info("No market data available.")
        return

    cols = config.get("columns") or df.columns.tolist()
    cols = [c for c in cols if c in df.columns]
    if not cols:
        st.warning("Select at least one column to display.")
        return

    sort_column = config.get("sort_column")
    if sort_column not in df.columns:
        sort_column = cols[0]

    df_view = df.sort_values(sort_column, ascending=not config.get("descending", True))
    df_view = df_view.head(config.get("top_n", 10))
    st.dataframe(df_view[cols], use_container_width=True, hide_index=True)


def _configure_alerts(context: DashboardContext, config: Dict[str, Any]) -> Dict[str, Any]:
    metric_options = {
        "price_change_percentage_24h": "%Δ 24h",
        "price_change_percentage_7d_in_currency": "%Δ 7d",
        "market_cap": "Market cap",
    }
    metric = st.selectbox(
        "Metric",
        options=list(metric_options.keys()),
        format_func=lambda key: metric_options[key],
        index=list(metric_options.keys()).index(config.get("metric", "price_change_percentage_24h")),
        key=f"alert_metric_{id(context)}",
    )
    direction = st.radio(
        "Direction",
        options=["above", "below"],
        index=0 if config.get("direction", "above") == "above" else 1,
        key=f"alert_dir_{id(context)}",
        horizontal=True,
    )
    threshold = st.number_input(
        "Threshold",
        value=float(config.get("threshold", 10.0)),
        key=f"alert_thresh_{id(context)}",
    )
    return {**config, "metric": metric, "direction": direction, "threshold": threshold}


def _render_alerts(context: DashboardContext, config: Dict[str, Any]) -> None:
    df = context.markets
    if df.empty:
        st.info("No market data available for alerts.")
        return

    metric = config.get("metric", "price_change_percentage_24h")
    if metric not in df.columns:
        st.warning("Selected metric not present in dataset.")
        return

    direction = config.get("direction", "above")
    threshold = config.get("threshold", 10.0)
    if direction == "above":
        triggered = df[df[metric] >= threshold]
        descriptor = f"≥ {threshold}"
    else:
        triggered = df[df[metric] <= threshold]
        descriptor = f"≤ {threshold}"

    label = metric.replace("_", " ").title()
    if triggered.empty:
        st.success(f"No coins found with {label} {descriptor}.")
        return

    st.warning(f"Coins with {label} {descriptor}:")
    st.dataframe(triggered[["name", "symbol", metric]], hide_index=True, use_container_width=True)


def _configure_portfolio(context: DashboardContext, config: Dict[str, Any]) -> Dict[str, Any]:
    if context.portfolio.empty:
        st.info("No portfolio dataset available.")
        return config
    view = st.selectbox(
        "Visualisation",
        options=["Bar", "Table"],
        index=0 if config.get("view", "Bar") == "Bar" else 1,
        key=f"portfolio_view_{id(context)}",
    )
    return {**config, "view": view}


def _render_portfolio(context: DashboardContext, config: Dict[str, Any]) -> None:
    df = context.portfolio
    if df.empty:
        st.info("Upload a portfolio dataset to unlock this widget.")
        return

    df = df.copy()
    df["allocation_pct"] = pd.to_numeric(df.get("allocation_pct"), errors="coerce")
    df["current_value"] = pd.to_numeric(df.get("current_value"), errors="coerce")
    df.dropna(subset=["holding_name", "current_value"], inplace=True)

    view = config.get("view", "Bar")
    if view == "Table":
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    fig = px.bar(
        df,
        x="holding_name",
        y="current_value",
        hover_data=["coin_id", "allocation_pct", "cost_basis"],
        template="plotly_white",
        color="holding_name",
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Holding",
        yaxis_title="Current Value",
        margin=dict(l=20, r=20, t=40, b=20),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)


WIDGETS: Dict[str, WidgetDefinition] = {
    "market_kpis": WidgetDefinition(
        id="market_kpis",
        label="Market KPIs",
        description="Aggregated insight cards for the selected universe of coins.",
        default_config={"metrics": ["total_market_cap", "volume_24h", "avg_change_24h"]},
        configure=_configure_market_kpis,
        render=_render_market_kpis,
    ),
    "price_chart": WidgetDefinition(
        id="price_chart",
        label="Price History",
        description="Interactive time series chart for the chosen assets.",
        default_config={"coins": [], "chart_type": "line", "normalize": False, "full_width": True},
        configure=_configure_price_chart,
        render=_render_price_chart,
    ),
    "market_table": WidgetDefinition(
        id="market_table",
        label="Market Table",
        description="Sortable table with flexible column selection.",
        default_config={"columns": [], "top_n": 10, "sort_column": "market_cap", "descending": True},
        configure=_configure_market_table,
        render=_render_market_table,
    ),
    "alerts": WidgetDefinition(
        id="alerts",
        label="Signals & Alerts",
        description="Highlight coins breaching KPI thresholds.",
        default_config={"metric": "price_change_percentage_24h", "direction": "above", "threshold": 10.0},
        configure=_configure_alerts,
        render=_render_alerts,
    ),
    "portfolio": WidgetDefinition(
        id="portfolio",
        label="Portfolio Overview",
        description="Visualise allocations from a supplementary dataset.",
        default_config={"view": "Bar"},
        configure=_configure_portfolio,
        render=_render_portfolio,
    ),
}

DEFAULT_LAYOUT = {
    "columns": 2,
    "widgets": [
        {"id": "market_kpis", "config": WIDGETS["market_kpis"].default_config.copy()},
        {"id": "price_chart", "config": WIDGETS["price_chart"].default_config.copy()},
        {"id": "market_table", "config": WIDGETS["market_table"].default_config.copy()},
        {"id": "alerts", "config": WIDGETS["alerts"].default_config.copy()},
    ],
}


# -----------------------------------------------------------------------------
# Load data from storage layers
# -----------------------------------------------------------------------------
markets = _load_table_cached(str(repository.db_path), "markets")
history = _load_table_cached(str(repository.db_path), "history")

portfolio = pd.DataFrame()
for dataset in config.get("supplementary_datasets", []):
    if dataset.get("id") == "portfolio":
        if dataset.get("type") == "csv":
            portfolio = _load_csv_cached(dataset.get("path", ""))
        else:
            try:
                portfolio = repository.load_dataset("portfolio")
            except KeyError:
                portfolio = pd.DataFrame()
        break

if "ts" in history.columns:
    history["ts"] = pd.to_datetime(history["ts"], errors="coerce")

# Normalise column casing
if "coin_id" not in history.columns and "id" in history.columns:
    history = history.rename(columns={"id": "coin_id"})

# -----------------------------------------------------------------------------
# Sidebar – global filters and dashboard configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    if markets.empty:
        st.warning("No market data found – run the ETL pipeline to populate the warehouse.")
    else:
        markets = markets.copy()
        markets["name"] = markets.get("name", markets.get("id", "")).astype(str)
        markets["symbol"] = markets.get("symbol", "").astype(str)

    available_coins = markets.get("id", pd.Series(dtype=str)).dropna().tolist()
    default_selection = available_coins[: min(5, len(available_coins))]
    selected_coins = st.multiselect(
        "Coins",
        options=available_coins,
        default=default_selection,
    )

    min_market_cap = float(markets.get("market_cap", pd.Series([0.0])).fillna(0).min()) if not markets.empty else 0.0
    max_market_cap = float(markets.get("market_cap", pd.Series([0.0])).fillna(0).max()) if not markets.empty else 0.0
    market_cap_range = st.slider(
        "Market cap range",
        min_value=0.0,
        max_value=max(1.0, max_market_cap),
        value=(min_market_cap, max_market_cap or 1.0),
        step=max((max_market_cap or 1.0) / 100, 1.0),
    )

    change_range = st.slider(
        "24h change (%)",
        min_value=-100.0,
        max_value=100.0,
        value=(-100.0, 100.0),
        step=1.0,
    )

    if not history.empty and history["ts"].notnull().any():
        min_date = history["ts"].min().date()
        max_date = history["ts"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    else:
        date_range = None

    st.markdown("---")
    st.header("Dashboard layout")

    if "dashboard_layout" not in st.session_state:
        st.session_state["dashboard_layout"] = json.loads(json.dumps(DEFAULT_LAYOUT))

    layout = st.session_state["dashboard_layout"]

    num_columns = st.slider(
        "Columns",
        min_value=1,
        max_value=3,
        value=layout.get("columns", 2),
    )
    layout["columns"] = num_columns

    selected_widget_ids = st.multiselect(
        "Widgets",
        options=list(WIDGETS.keys()),
        format_func=lambda key: WIDGETS[key].label,
        default=[w["id"] for w in layout.get("widgets", [])],
    )

    current_widgets = {w["id"]: w for w in layout.get("widgets", [])}
    new_widget_list = []
    for widget_id in selected_widget_ids:
        if widget_id in current_widgets:
            new_widget_list.append(current_widgets[widget_id])
        else:
            new_widget_list.append({"id": widget_id, "config": WIDGETS[widget_id].default_config.copy()})
    layout["widgets"] = new_widget_list

    st.subheader("Widget configuration")
    context_preview = DashboardContext(
        markets=markets,
        history=history,
        portfolio=portfolio,
        filters={"available_coins": available_coins, "selected_coins": selected_coins},
    )
    for widget_entry in layout["widgets"]:
        widget = WIDGETS[widget_entry["id"]]
        with st.expander(widget.label, expanded=False):
            widget_entry["config"] = widget.configure(context_preview, widget_entry.get("config", {}))

    st.markdown("---")
    st.subheader("Persistence")

    existing_dashboards = sorted(dashboards_dir.glob("*.json"))
    load_choice = st.selectbox(
        "Load preset",
        options=["-- Select --"] + [f.stem for f in existing_dashboards],
    )
    if st.button("Load layout") and load_choice != "-- Select --":
        file_path = dashboards_dir / f"{load_choice}.json"
        try:
            loaded_layout = json.loads(file_path.read_text(encoding="utf-8"))
            st.session_state["dashboard_layout"] = loaded_layout
            st.success(f"Loaded dashboard '{load_choice}'.")
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Failed to load dashboard: {exc}")

    save_name = st.text_input("Save as", value="My dashboard")
    if st.button("Save layout"):
        file_name = _slugify(save_name) + ".json"
        target = dashboards_dir / file_name
        target.write_text(json.dumps(layout, indent=2), encoding="utf-8")
        st.success(f"Dashboard saved as '{file_name}'.")

    if st.button("Reset to default"):
        st.session_state["dashboard_layout"] = json.loads(json.dumps(DEFAULT_LAYOUT))
        st.success("Dashboard reset.")


# -----------------------------------------------------------------------------
# Apply filters to datasets
# -----------------------------------------------------------------------------
filtered_markets = markets.copy()
if not filtered_markets.empty:
    if selected_coins:
        filtered_markets = filtered_markets[filtered_markets["id"].isin(selected_coins)]
    mc_min, mc_max = market_cap_range
    if "market_cap" in filtered_markets:
        filtered_markets = filtered_markets[
            filtered_markets["market_cap"].fillna(0).between(mc_min, mc_max)
        ]
    ch_min, ch_max = change_range
    if "price_change_percentage_24h" in filtered_markets:
        filtered_markets = filtered_markets[
            filtered_markets["price_change_percentage_24h"].fillna(0).between(ch_min, ch_max)
        ]

filtered_history = history.copy()
if not filtered_history.empty:
    if selected_coins:
        filtered_history = filtered_history[filtered_history["coin_id"].isin(selected_coins)]
    if date_range and len(date_range) == 2:
        tzinfo = filtered_history["ts"].dt.tz if pd.api.types.is_datetime64tz_dtype(filtered_history["ts"]) else None
        if tzinfo is not None:
            start_date = pd.Timestamp(date_range[0]).tz_localize(tzinfo, nonexistent="shift_forward", ambiguous="NaT")
            end_date = (
                pd.Timestamp(date_range[1]).tz_localize(tzinfo, nonexistent="shift_forward", ambiguous="NaT")
                + pd.Timedelta(days=1)
                - pd.Timedelta(seconds=1)
            )
        else:
            start_date = pd.Timestamp(date_range[0])
            end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered_history = filtered_history[
            (filtered_history["ts"] >= start_date) & (filtered_history["ts"] <= end_date)
        ]

filtered_portfolio = portfolio.copy()
if not filtered_portfolio.empty and selected_coins:
    filtered_portfolio = filtered_portfolio[filtered_portfolio["coin_id"].isin(selected_coins)]

context = DashboardContext(
    markets=filtered_markets,
    history=filtered_history,
    portfolio=filtered_portfolio,
    filters={
        "selected_coins": selected_coins,
        "available_coins": available_coins,
        "date_range": date_range,
        "market_cap_range": market_cap_range,
        "change_range": change_range,
    },
)


# -----------------------------------------------------------------------------
# Render dashboard widgets
# -----------------------------------------------------------------------------
layout = st.session_state.get("dashboard_layout", DEFAULT_LAYOUT)
widgets = layout.get("widgets", [])
num_columns = layout.get("columns", 2)
columns = st.columns(num_columns) if num_columns > 1 else [st]

for index, widget_entry in enumerate(widgets):
    widget = WIDGETS.get(widget_entry["id"])
    if not widget:
        continue

    config = widget_entry.get("config", widget.default_config.copy())
    container = st.container() if config.get("full_width") or num_columns == 1 else columns[index % num_columns]
    with container:
        st.subheader(widget.label)
        st.caption(widget.description)
        widget.render(context, config)
        st.markdown("---")
