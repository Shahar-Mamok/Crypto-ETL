import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crypto Market Dashboard", layout="wide")

st.title("💹 **Crypto Market Dashboard**")
st.title("Crypto Market Dashboard")
st.caption("Powered by CoinGecko | Data refreshed via ETL pipeline")

@st.cache_data(ttl=300)
def load_table(table: str) -> pd.DataFrame:
    conn = sqlite3.connect("data/warehouse.db")
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

markets = load_table("markets")
history = load_table("history")
history["ts"] = pd.to_datetime(history["ts"], format='mixed')

# === Sidebar Filters ===
st.sidebar.header("🔎 Filters")
coins_selected = st.sidebar.multiselect(
    "Select Coins:", markets["id"].tolist(), default=["bitcoin", "ethereum"]
)
date_range = st.sidebar.slider(
    "Select Date Range:",
    min_value=pd.to_datetime(history["ts"]).min().date(),
    max_value=pd.to_datetime(history["ts"]).max().date(),
    value=(
        pd.to_datetime(history["ts"]).max().date() - pd.Timedelta(days=7),
        pd.to_datetime(history["ts"]).max().date()
    )
)

# Filter data
hist_filtered = history[
    (history["coin_id"].isin(coins_selected)) &
    (history["ts"].dt.date >= date_range[0]) &
    (history["ts"].dt.date <= date_range[1])
]

# === KPI Section ===
st.subheader("📊 Market Snapshot")
cols = st.columns(len(coins_selected))
for i, coin in enumerate(coins_selected):
    data = markets[markets["id"] == coin].iloc[0]
    cols[i].metric(
        label=f"{data['name']} ({data['symbol'].upper()})",
        value=f"${data['current_price']:,}",
        delta=f"{data['price_change_percentage_24h']:.2f}% (24h)"
    )

# === Charts ===
st.subheader("📈 Price History")
for coin in coins_selected:
    df_coin = hist_filtered[hist_filtered["coin_id"] == coin]
    st.line_chart(df_coin.set_index("ts")["price"], height=300)

# === Top Coins Table ===
st.subheader("🏆 Top 10 Coins by Market Cap")
top10 = markets.sort_values("market_cap", ascending=False).head(10)
st.dataframe(
    top10[["name", "symbol", "current_price", "market_cap", "total_volume"]]
    .rename(columns={
        "name": "Name",
        "symbol": "Symbol",
        "current_price": "Current Price",
        "market_cap": "Market Cap",
        "total_volume": "Total Volume"
    }),
    use_container_width=True
)
