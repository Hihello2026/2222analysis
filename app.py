import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np
from datetime import datetime
import requests

# 1. Dashboard UI Configuration
st.set_page_config(page_title="Archer Matrix | archb26", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; color: #1a1a1a; }
    [data-testid="stMetricValue"] { color: #0f172a; font-weight: 800; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 4px;
        border: 1px solid #e2e8f0;
    }
    div[data-testid="stTable"] { background-color: #ffffff; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

st.title("Strategic Asset Allocation")
st.markdown(f"**Archive Date:** {datetime.now().strftime('%Y-%m-%d')} | **Status:** Moat Optimized")

# 2. Strategy Mapping (BSF & Diversified Moats)
moat_assets = {
    '2222.SR': {'name': 'Aramco', 'moat': 'Cost Leadership', 'yield': 0.065},
    '2223.SR': {'name': 'Luberef', 'moat': 'Base Oil Specialist', 'yield': 0.072},
    '2083.SR': {'name': 'Marafiq', 'moat': 'Utility Monopoly', 'yield': 0.038},
    '5110.SR': {'name': 'SEC', 'moat': 'National Grid', 'yield': 0.040},
    '1111.SR': {'name': 'Tadawul', 'moat': 'Sole Operator', 'yield': 0.028},
    '1050.SR': {'name': 'BSF', 'moat': 'Corporate Banking Lead', 'yield': 0.045},
    '1180.SR': {'name': 'SNB', 'moat': 'Giga-Project Capital', 'yield': 0.039},
    '8313.SR': {'name': 'Rasan', 'moat': 'Network Effect', 'yield': 0.012},
    '7217.SR': {'name': 'ELM', 'moat': 'Exclusive Data Integration', 'yield': 0.015},
    '2381.SR': {'name': 'Arabian Drilling', 'moat': 'Critical Infrastructure', 'yield': 0.035},
    '7010.SR': {'name': 'stc', 'moat': 'Digital Backbone', 'yield': 0.052},
    '4030.SR': {'name': 'Bahri', 'moat': 'Maritime Logistics Lead', 'yield': 0.048},
    '4263.SR': {'name': 'SAL', 'moat': 'Air Cargo Monopoly', 'yield': 0.022},
    '4031.SR': {'name': 'SGS', 'moat': 'Airport Operations', 'yield': 0.042},
    '4013.SR': {'name': 'HMG', 'moat': 'Premium Healthcare Efficiency', 'yield': 0.019},
    '1211.SR': {'name': 'Maaden', 'moat': 'Resource Rights', 'yield': 0.000},
    '2020.SR': {'name': 'SABIC AN', 'moat': 'Production Efficiency', 'yield': 0.044},
    '2280.SR': {'name': 'Almarai', 'moat': 'Distribution Power', 'yield': 0.025},
    '1830.SR': {'name': 'Leejam', 'moat': 'Fitness Scale', 'yield': 0.024}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. Data Core
@st.cache_data
def get_live_data(symbols, start, end):
    try:
        data = yf.download(symbols + ['^TASI.SR'], start=start, end=end, progress=False)['Close']
        data = data.ffill().dropna(axis=1, thresh=len(data)*0.5).dropna()
        benchmark = data['^TASI.SR']
        assets = data.drop(columns=['^TASI.SR']).rename(columns=mapping)
        return assets, benchmark
    except:
        return pd.DataFrame(), pd.Series()

# Sidebar
st.sidebar.header("Backtest Configuration")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-06-15"))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
capital = st.sidebar.number_input("Total Capital (SAR)", value=1000000)

# Telegram Messenger
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={msg}"
    try: requests.get(url)
    except: pass

# 4. Analytics Output
if start_date < end_date:
    assets_data, tasi_data = get_live_data(tickers, start_date, end_date)

    if not assets_data.empty:
        # Optimization
        mu = expected_returns.mean_historical_return(assets_data)
        S = risk_models.CovarianceShrinkage(assets_data).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.09))
        weights = ef.max_sharpe(risk_free_rate=0.04)
        clean_weights = ef.clean_weights()

        # KPIs
        p_ret, p_vol, p_sharpe = ef.portfolio_performance(risk_free_rate=0.04)
        
        st.subheader("Performance & Dividend Intelligence")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Capital Return", f"{p_ret:.2%}")
        
        total_div = sum([clean_weights.get(mapping[t], 0) * moat_assets[t]['yield'] for t in tickers if mapping[t] in clean_weights])
        m2.metric("Portfolio Yield", f"{total_div:.2%}")
        m3.metric("Annual Div. Income", f"{(total_div * capital):,.2f} SAR")
        m4.metric("Sharpe Ratio", f"{p_sharpe:.2f}")

        # --- Updated Section Header: Alerts ---
        st.markdown("---")
        st.subheader("Alerts")
        
        target_configs = {
            'Aramco': 28.50, 'stc': 36.00, 'BSF': 32.00, 
            'Luberef': 130.00, 'HMG': 270.00
        }
        
        alert_cols = st.columns(len(target_configs))
        for i, (name, target) in enumerate(target_configs.items()):
            if name in assets_data.columns:
                price = assets_data[name].iloc[-1]
                with alert_cols[i]:
                    if price <= target:
                        st.error(f"🚨 BUY: {name}")
                        st.write(f"Price: {price:.2f}")
                        if st.button(f"Notify {name}"):
                            send_telegram(f"Archer Alert: {name} is at {price:.2f} (Target: {target})")
                            st.toast("Sent!")
                    else:
                        st.success(f"✅ {name}")
                        st.caption(f"Price: {price:.2f}")

        # Detailed Stats
        st.markdown("---")
        st.subheader("Detailed Asset Allocation")
        alloc_data = []
        for t in tickers:
            w = clean_weights.get(mapping[t], 0)
            if w > 0:
                alloc_data.append({
                    "Asset": mapping[t],
                    "Moat Strategy": moat_assets[t]['moat'],
                    "Weight": f"{w:.2%}",
                    "Div. Yield": f"{moat_assets[t]['yield']:.2%}",
                    "Est. Income": f"{(w * capital * moat_assets[t]['yield']):,.2f} SAR"
                })
        st.table(pd.DataFrame(alloc_data))

        # Charting
        st.markdown("---")
        st.subheader("Growth Chart: Portfolio vs. TASI")
        p_daily = assets_data.pct_change().dropna().dot(np.array([clean_weights.get(c, 0) for c in assets_data.columns]))
        st.line_chart(pd.DataFrame({
            'Portfolio': (1 + p_daily).cumprod(),
            'TASI Index': (1 + tasi_data.pct_change().dropna()).cumprod()
        }))
