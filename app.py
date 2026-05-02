import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np
from datetime import datetime
import requests

# 1. إعدادات الهوية والتصميم
st.set_page_config(page_title="Archer Matrix | archb26", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    [data-testid="stMetricValue"] { color: #0f172a; font-weight: 800; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 4px; border: 1px solid #e2e8f0; }
    .blue-card {
        background-color: #e0f2fe;
        border-left: 5px solid #0284c7;
        padding: 15px;
        border-radius: 5px;
        color: #0369a1;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Strategic Asset Allocation")
st.markdown(f"**Archive Date:** {datetime.now().strftime('%Y-%m-%d')} | **Status:** Moat Optimized")

# 2. تكوين المحفظة (19 شركة)
moat_assets = {
    '2222.SR': {'name': 'Aramco', 'moat': 'Cost Leadership', 'yield': 0.065, 'target': 28.50},
    '2223.SR': {'name': 'Luberef', 'moat': 'Base Oil Specialist', 'yield': 0.072, 'target': 130.00},
    '1050.SR': {'name': 'BSF', 'moat': 'Corporate Banking Lead', 'yield': 0.045, 'target': 32.00},
    '7010.SR': {'name': 'stc', 'moat': 'Digital Backbone', 'yield': 0.052, 'target': 36.00},
    '4013.SR': {'name': 'HMG', 'moat': 'Premium Healthcare Efficiency', 'yield': 0.019, 'target': 270.00},
    '1180.SR': {'name': 'SNB', 'moat': 'Giga-Project Capital', 'yield': 0.039, 'target': 34.00},
    '1111.SR': {'name': 'Tadawul', 'moat': 'Sole Operator', 'yield': 0.028, 'target': 180.00},
    '5110.SR': {'name': 'SEC', 'moat': 'National Grid', 'yield': 0.040, 'target': 19.50},
    '2083.SR': {'name': 'Marafiq', 'moat': 'Utility Monopoly', 'yield': 0.038, 'target': 62.00},
    '4030.SR': {'name': 'Bahri', 'moat': 'Maritime Logistics Lead', 'yield': 0.048, 'target': 24.00},
    '4263.SR': {'name': 'SAL', 'moat': 'Air Cargo Monopoly', 'yield': 0.022, 'target': 230.00},
    '4031.SR': {'name': 'SGS', 'moat': 'Airport Operations', 'yield': 0.042, 'target': 33.00},
    '8313.SR': {'name': 'Rasan', 'moat': 'Network Effect', 'yield': 0.012, 'target': 40.00},
    '7217.SR': {'name': 'ELM', 'moat': 'Exclusive Data Integration', 'yield': 0.015, 'target': 900.00},
    '2381.SR': {'name': 'Arabian Drilling', 'moat': 'Critical Infrastructure', 'yield': 0.035, 'target': 140.00},
    '1211.SR': {'name': 'Maaden', 'moat': 'Resource Rights', 'yield': 0.000, 'target': 45.00},
    '2020.SR': {'name': 'SABIC AN', 'moat': 'Production Efficiency', 'yield': 0.044, 'target': 65.00},
    '2280.SR': {'name': 'Almarai', 'moat': 'Distribution Power', 'yield': 0.025, 'target': 52.00},
    '1830.SR': {'name': 'Leejam', 'moat': 'Fitness Scale', 'yield': 0.024, 'target': 190.00}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. إعدادات التيليجرام
BOT_TOKEN = "8096609350:AAGDqysumgrjhlcOniM6D885j620QqCWpc8"

def send_telegram(msg, chat_id):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
    try: requests.post(url, data=payload)
    except: pass

# 4. معالجة البيانات
@st.cache_data(ttl=3600)
def get_live_data(symbols, start, end):
    try:
        raw = yf.download(symbols + ['^TASI.SR'], start=start, end=end, progress=False)['Close']
        data = raw.ffill().dropna()
        if data.empty: return pd.DataFrame(), pd.Series()
        benchmark = data['^TASI.SR']
        assets = data.drop(columns=['^TASI.SR']).rename(columns=mapping)
        return assets, benchmark
    except: return pd.DataFrame(), pd.Series()

# Sidebar
st.sidebar.header("Configuration")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-06-15"))
capital = st.sidebar.number_input("Total Capital (SAR)", value=1000000)
user_id = st.sidebar.text_input("Telegram ID", value="7172975999")

st.sidebar.markdown("---")
st.sidebar.warning("""
**Disclaimer / إخلاء مسؤولية**
ليست دعوة للاستثمار. قرارك مسؤوليتك.
Not financial advice. Invest at your own risk.
""")

# 5. التنفيذ والعرض
assets_data, tasi_data = get_live_data(tickers, start_date, datetime.now())

if not assets_data.empty:
    # التحسين (Optimization)
    mu = expected_returns.mean_historical_return(assets_data)
    S = risk_models.CovarianceShrinkage(assets_data).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.09))
    weights = ef.max_sharpe(risk_free_rate=0.04)
    clean_weights = ef.clean_weights()
    p_ret, p_vol, p_sharpe = ef.portfolio_performance(risk_free_rate=0.04)
    
    # المقاييس الرئيسية
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected Return", f"{p_ret:.2%}")
    total_yield = sum([clean_weights.get(mapping[t], 0) * moat_assets[t]['yield'] for t in tickers])
    m2.metric("Portfolio Yield", f"{total_yield:.2%}")
    m3.metric("Est. Annual Div.", f"{(total_yield * capital):,.0f} SAR")
    m4.metric("Sharpe Ratio", f"{p_sharpe:.2f}")

    # التنبيهات (Alerts)
    st.markdown("---")
    st.subheader("Moat Alerts")
    alert_names = ['Aramco', 'stc', 'BSF', 'Luberef', 'HMG']
    cols = st.columns(len(alert_names))
    for i, name in enumerate(alert_names):
        price = assets_data[name].iloc[-1]
        ticker_id = [k for k, v in mapping.items() if v == name][0]
        target = moat_assets[ticker_id]['target']
        with cols[i]:
            if price <= target:
                st.error(f"🚨 BUY: {name}")
                if st.button(f"Notify {name}", key=name):
                    msg = f"*Archer Alert* 🎯\nAsset: {name}\nPrice: {price:.2f}\nTarget: {target:.2f}"
                    send_telegram(msg, user_id)
                    st.toast("Sent!")
            else:
                st.markdown(f'<div class="blue-card"><strong>✅ {name}</strong><br>Price: {price:.2f}</div>', unsafe_allow_html=True)

    # الجداول والرسوم
    st.markdown("---")
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("Allocation")
        alloc = [{"Asset": n, "Weight": f"{w:.2%}"} for n, w in clean_weights.items() if w > 0]
        st.table(pd.DataFrame(alloc))
    with col_right:
        st.subheader("Performance vs TASI")
        p_daily = assets_data.pct_change().dropna().dot(np.array([clean_weights.get(c, 0) for c in assets_data.columns]))
        st.line_chart(pd.DataFrame({'Portfolio': (1 + p_daily).cumprod(), 'TASI': (1 + tasi_data.pct_change().dropna()).cumprod()}))
else:
    st.warning("⚠️ Market data unavailable. Please check internet connection or refresh.")
