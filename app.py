import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np
from datetime import datetime
import requests

# 1. Branding & UI Setup
st.set_page_config(page_title="Archer Matrix | archb26", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    [data-testid="stMetricValue"] { color: #0f172a; font-weight: 800; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 4px; border: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

st.title("Strategic Asset Allocation")
st.markdown(f"**Archive Date:** {datetime.now().strftime('%Y-%m-%d')} | **Status:** Moat Optimized")

# 2. Portfolio Configuration (The 19 Strategic Assets)
moat_assets = {
    '2222.SR': {'name': 'Aramco', 'moat': 'Cost Leadership', 'yield': 0.065},
    '2223.SR': {'name': 'Luberef', 'moat': 'Base Oil Specialist', 'yield': 0.072},
    '1050.SR': {'name': 'BSF', 'moat': 'Corporate Banking Lead', 'yield': 0.045},
    '7010.SR': {'name': 'stc', 'moat': 'Digital Backbone', 'yield': 0.052},
    '4013.SR': {'name': 'HMG', 'moat': 'Premium Healthcare Efficiency', 'yield': 0.019},
    '1180.SR': {'name': 'SNB', 'moat': 'Giga-Project Capital', 'yield': 0.039},
    '1111.SR': {'name': 'Tadawul', 'moat': 'Sole Operator', 'yield': 0.028},
    '5110.SR': {'name': 'SEC', 'moat': 'National Grid', 'yield': 0.040},
    '2083.SR': {'name': 'Marafiq', 'moat': 'Utility Monopoly', 'yield': 0.038},
    '4030.SR': {'name': 'Bahri', 'moat': 'Maritime Logistics Lead', 'yield': 0.048},
    '4263.SR': {'name': 'SAL', 'moat': 'Air Cargo Monopoly', 'yield': 0.022},
    '4031.SR': {'name': 'SGS', 'moat': 'Airport Operations', 'yield': 0.042},
    '8313.SR': {'name': 'Rasan', 'moat': 'Network Effect', 'yield': 0.012},
    '7217.SR': {'name': 'ELM', 'moat': 'Exclusive Data Integration', 'yield': 0.015},
    '2381.SR': {'name': 'Arabian Drilling', 'moat': 'Critical Infrastructure', 'yield': 0.035},
    '1211.SR': {'name': 'Maaden', 'moat': 'Resource Rights', 'yield': 0.000},
    '2020.SR': {'name': 'SABIC AN', 'moat': 'Production Efficiency', 'yield': 0.044},
    '2280.SR': {'name': 'Almarai', 'moat': 'Distribution Power', 'yield': 0.025},
    '1830.SR': {'name': 'Leejam', 'moat': 'Fitness Scale', 'yield': 0.024}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. Telegram Messaging Core
BOT_TOKEN = "8096609350:AAGDqysumgrjhlcOniM6D885j620QqCWpc8"
CHAT_ID = "7172975999"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except:
        pass

# 4. Data Processing
@st.cache_data
def get_live_data(symbols, start, end):
    try:
        data = yf.download(symbols + ['^TASI.SR'], start=start, end=end, progress=False)['Close']
        data = data.ffill().dropna()
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

# Disclaimer in Sidebar
st.sidebar.markdown("---")
st.sidebar.warning("""
**إخلاء مسؤولية / Disclaimer**
هذه المنصة لأغراض تعليمية فقط وليست دعوة للاستثمار.
Educational purposes only; not an investment invitation.
""")

# 5. UI Logic and Output
if start_date < end_date:
    assets_data, tasi_data = get_live_data(tickers, start_date, end_date)

    if not assets_data.empty:
        # Optimization
        mu = expected_returns.mean_historical_return(assets_data)
        S = risk_models.CovarianceShrinkage(assets_data).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.09))
        weights = ef.max_sharpe(risk_free_rate=0.04)
        clean_weights = ef.clean_weights()
        p_ret, p_vol, p_sharpe = ef.portfolio_performance(risk_free_rate=0.04)
        
        # Metrics Header
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Capital Return", f"{p_ret:.2%}")
        total_div = sum([clean_weights.get(mapping[t], 0) * moat_assets[t]['yield'] for t in tickers if mapping[t] in clean_weights])
        m2.metric("Portfolio Yield", f"{total_div:.2%}")
        m3.metric("Annual Div. Income", f"{(total_div * capital):,.2f} SAR")
        m4.metric("Sharpe Ratio", f"{p_sharpe:.2f}")

        # Alerts with Telegram Trigger
        st.markdown("---")
        st.subheader("Alerts")
        target_configs = {'Aramco': 28.50, 'stc': 36.00, 'BSF': 32.00, 'Luberef': 130.00, 'HMG': 270.00}
        
        alert_cols = st.columns(len(target_configs))
        for i, (name, target) in enumerate(target_configs.items()):
            if name in assets_data.columns:
                price = assets_data[name].iloc[-1]
                with alert_cols[i]:
                    if price <= target:
                        st.error(f"🚨 BUY: {name}")
                        if st.button(f"Notify {name}"):
                            msg = f"*Archer Alert* 🎯\n\nAsset: {name}\nPrice: {price:.2f}\nTarget: {target:.2f}\nStatus: Buy Zone"
                            send_telegram(msg)
                            st.toast("Telegram Sent!")
                    else:
                        st.success(f"✅ {name}")
                        st.caption(f"Price: {price:.2f}")

        # Tables & Charts
        st.markdown("---")
        st.subheader("Detailed Asset Allocation")
        alloc_data = [{"Asset": mapping[t], "Moat": moat_assets[t]['moat'], "Weight": f"{clean_weights.get(mapping[t], 0):.2%}"} for t in tickers if clean_weights.get(mapping[t], 0) > 0]
        st.table(pd.DataFrame(alloc_data))

        p_daily = assets_data.pct_change().dropna().dot(np.array([clean_weights.get(c, 0) for c in assets_data.columns]))
        st.line_chart(pd.DataFrame({'Portfolio': (1 + p_daily).cumprod(), 'TASI': (1 + tasi_data.pct_change().dropna()).cumprod()}))
