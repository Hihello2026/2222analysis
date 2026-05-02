import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

# 1. إعدادات الهوية والتصميم (Modern Industrial Minimalism)
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
# استخدام تاريخ آخر إغلاق متاح لضمان استقرار العرض
st.markdown(f"**Archive Date:** {datetime.now().strftime('%Y-%m-%d')} | **Status:** Moat Optimized")

# 2. بيانات الشركات الـ 19 (The Strategic Core)
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

# 4. محرك البيانات المحدث (Data Engine)
@st.cache_data(ttl=3600)
def load_market_data(symbols, start):
    try:
        # طلب البيانات لغاية تاريخ اليوم
        raw = yf.download(symbols + ['^TASI.SR'], start=start, progress=False)['Close']
        if raw.empty: return pd.DataFrame(), pd.Series()
        
        # معالجة بيانات أيام العطلات (ffill تملأ الفراغات بآخر سعر متاح)
        data = raw.ffill().dropna()
        benchmark = data['^TASI.SR']
        assets = data.drop(columns=['^TASI.SR']).rename(columns=mapping)
        return assets, benchmark
    except:
        return pd.DataFrame(), pd.Series()

# Sidebar
st.sidebar.header("Archer Config")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-06-15"))
capital = st.sidebar.number_input("Capital (SAR)", value=1000000)
tg_id = st.sidebar.text_input("Telegram ID", value="7172975999")

st.sidebar.markdown("---")
st.sidebar.warning("**Disclaimer:** Educational purposes only.")

# 5. التنفيذ والعرض (Logic Flow)
assets_data, tasi_data = load_market_data(tickers, start_date)

if not assets_data.empty:
    # حسابات المحفظة
    mu = expected_returns.mean_historical_return(assets_data)
    S = risk_models.CovarianceShrinkage(assets_data).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.09))
    weights = ef.max_sharpe(risk_free_rate=0.04)
    clean_weights = ef.clean_weights()
    p_ret, p_vol, p_sharpe = ef.portfolio_performance(risk_free_rate=0.04)
    
    # ملخص الأداء
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exp. Return", f"{p_ret:.2%}")
    p_yield = sum([clean_weights.get(mapping[t], 0) * moat_assets[t]['yield'] for t in tickers])
    c2.metric("Portfolio Yield", f"{p_yield:.2%}")
    c3.metric("Annual Div.", f"{(p_yield * capital):,.0f} SAR")
    c4.metric("Sharpe", f"{p_sharpe:.2f}")

    # التنبيهات (Alert Grid)
    st.markdown("---")
    st.subheader("Moat Alerts")
    watch_list = ['Aramco', 'stc', 'BSF', 'Luberef', 'HMG']
    cols = st.columns(5)
    for i, name in enumerate(watch_list):
        price = assets_data[name].iloc[-1]
        t_id = [k for k,v in mapping.items() if v==name][0]
        target = moat_assets[t_id]['target']
        with cols[i]:
            if price <= target:
                st.error(f"🚨 BUY: {name}")
                if st.button(f"Notify {name}", key=name):
                    msg = f"*Archer Alert* 🎯\nAsset: {name}\nPrice: {price:.2f}\nTarget: {target:.2f}"
                    send_telegram(msg, tg_id)
                    st.toast("Sent!")
            else:
                st.markdown(f'<div class="blue-card"><strong>✅ {name}</strong><br>Price: {price:.2f}</div>', unsafe_allow_html=True)

    # الرسوم البيانية
    st.markdown("---")
    p_daily = assets_data.pct_change().dropna().dot(np.array([clean_weights.get(c, 0) for c in assets_data.columns]))
    st.line_chart(pd.DataFrame({'Portfolio': (1 + p_daily).cumprod(), 'TASI': (1 + tasi_data.pct_change().dropna()).cumprod()}))
else:
    # في حال استمرار المشكلة (غالباً بسبب اتصال السيرفر بـ Yahoo Finance)
    st.error("⚠️ Connection Error: Yahoo Finance is not responding. Please try again in 5 minutes.")
