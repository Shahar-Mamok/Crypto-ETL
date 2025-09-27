import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------------------
# Page config & basic styles
# ----------------------------------
st.set_page_config(page_title="Crypto Market Dashboard", layout="wide", page_icon="💹")

st.markdown("""
<style>
body, .main {
    background-color: #fff8f0;
}
.coin-card {
    background: #fff8f0;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(255,140,0,0.08);
    padding: 2rem 2.5rem 2rem 2.5rem;
    margin-bottom: 2rem;
    color: #ff6600;
}
.coin-header {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}
.coin-logo {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    border: 2px solid #ffe0b2;
    background: #fff;
    object-fit: contain;
}
.coin-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    color: #ff6600;
}
.coin-symbol {
    font-size: 1.2rem;
    color: #ff9800;
    font-weight: 500;
}
.coin-stats {
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.coin-stat-card {
    background: #fff3e0;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    min-width: 160px;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 4px rgba(255,140,0,0.04);
    color: #ff6600;
}
.coin-section-title {
    font-size: 1.3rem;
    font-weight: 600;
    margin: 1.5rem 0 0.7rem 0;
    color: #ff9800;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #    background: #fff3e0; !important;
}
/* Table header */
thead tr th {
    background-color: #ff9800 !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='margin-bottom:0.2rem; color:#ff6600;'> Crypto Market Dashboard</h1>", unsafe_allow_html=True)
st.caption(":rocket: <b style='color:#ff9800;'>Powered by CoinGecko | Data refreshed via ETL pipeline</b>", unsafe_allow_html=True)


# ----------------------------------
# Data loading
# ----------------------------------
@st.cache_data(ttl=300)
def load_table(table: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect("data/warehouse.db")
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Failed to load table '{table}' from SQLite. Error: {e}")
        return pd.DataFrame()

def normalize_markets(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "id", "name", "symbol", "image", "current_price", "market_cap", "total_volume",
        "price_change_percentage_24h", "ath", "ath_date", "atl", "atl_date",
        "circulating_supply", "total_supply", "max_supply"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df

# ----------------------------------
# Load data
# ----------------------------------
markets = load_table("markets")
history = load_table("history")
markets = normalize_markets(markets)

if markets.empty:
    st.error("No market data available. Please run the ETL pipeline.")
    st.stop()
if history.empty:
    st.warning("No price history data available.")

# ----------------------------------
# Sidebar: Coin selection
# ----------------------------------
with st.sidebar:
    st.header("Coins")
    coin_options = markets[["id", "name", "symbol", "image"]].copy()
    if "market_cap" in coin_options.columns:
        coin_options = coin_options.sort_values("market_cap", ascending=False)
    else:
        coin_options = coin_options.sort_values("name")
    selected_coin = st.selectbox(
        "Select a coin:",
        options=coin_options["id"].tolist(),
        format_func=lambda cid: f"{coin_options[coin_options['id']==cid]['name'].values[0]} ({coin_options[coin_options['id']==cid]['symbol'].values[0].upper()})"
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Select a coin to view its profile page.")

# ----------------------------------
# Main: Coin profile card
# ----------------------------------
coin = markets[markets["id"] == selected_coin].iloc[0]
coin_hist = history[history["coin_id"] == selected_coin] if not history.empty else pd.DataFrame()
coin_hist = coin_hist.sort_values("ts") if not coin_hist.empty else coin_hist

def safe_get(val, default="-"):
    return val if pd.notnull(val) else default

logo_url = safe_get(coin["image"])
symbol = safe_get(coin["symbol"]).upper()
name = safe_get(coin["name"])
latest_price = safe_get(coin["current_price"], 0)
price_change_24h = safe_get(coin["price_change_percentage_24h"], 0)
market_cap = safe_get(coin["market_cap"], 0)
volume = safe_get(coin["total_volume"], 0)
ath = safe_get(coin["ath"], 0)
ath_date = safe_get(coin["ath_date"])
atl = safe_get(coin["atl"], 0)
atl_date = safe_get(coin["atl_date"])
circulating = safe_get(coin["circulating_supply"], 0)
total_supply = safe_get(coin["total_supply"], 0)
max_supply = safe_get(coin["max_supply"], "∞")

coin_card_html = f"""
<div class='coin-card'>
<div class='coin-header'>
<img src='{logo_url}' class='coin-logo' alt='{name} logo'>
<div>
<div class='coin-title'>{name}</div>
<div class='coin-symbol'>{symbol}</div>
</div>
</div>
<div class='coin-stats'>
<div class='coin-stat-card'><b>Price</b><br>{latest_price:,.2f}</div>
<div class='coin-stat-card'><b>24h Change</b><br>{price_change_24h:+.2f}%</div>
<div class='coin-stat-card'><b>Market Cap</b><br>{market_cap:,.0f}</div>
<div class='coin-stat-card'><b>Volume (24h)</b><br>{volume:,.0f}</div>
<div class='coin-stat-card'><b>ATH</b><br>{ath:,.2f}</div>
<div class='coin-stat-card'><b>ATL</b><br>{atl:,.2f}</div>
<div class='coin-stat-card'><b>Circulating</b><br>{circulating:,.0f}</div>
<div class='coin-stat-card'><b>Total Supply</b><br>{total_supply:,.0f}</div>
<div class='coin-stat-card'><b>Max Supply</b><br>{max_supply if pd.notnull(max_supply) else '∞'}</div>
</div>
<div class='coin-section-title'>Price Chart</div>
<div>
<!-- Chart will be rendered below -->
</div>
</div>
"""
st.markdown(coin_card_html, unsafe_allow_html=True)

# ----------------------------------
# Price chart
# ----------------------------------
if not coin_hist.empty and 'price' in coin_hist.columns and coin_hist['price'].notnull().any():
    chart_col, _ = st.columns([3, 1])
    with chart_col:
        fig = px.line(
            coin_hist,
            x="ts",
            y="price",
            title=f"{name} Price Over Time",
            markers=True,
            template="plotly_white",
        )
        fig.update_traces(line=dict(width=2, color="#ff6600"))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="#fff8f0",
            paper_bgcolor="#fff8f0"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"No price data for {name} in selected range.")

# ----------------------------------
# About section (placeholder)
# ----------------------------------
about_html = f"""
<div class='coin-card'>
<div class='coin-section-title'>About {name}</div>
<p><b>{name}</b> ({symbol}) is a leading cryptocurrency. More info and unique features can be added here.</p>
</div>
"""
st.markdown(about_html, unsafe_allow_html=True)

# ----------------------------------
# Top coins table (optional)
# ----------------------------------
with st.expander("Top 10 Coins by Market Cap"):
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
        use_container_width=True,
        hide_index=True
    )
