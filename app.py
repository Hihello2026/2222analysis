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
    </style>
    """, unsafe_allow_html=True)

st.title("Strategic Asset Allocation")
st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')} | **Status:** Moat Portfolio Active")

# 2. تعريف بيانات الشركات (الخندق الاقتصادي والعوائد)
moat_assets = {
    '2222.SR': {'name': 'Aramco', 'moat': 'Cost Leadership', 'yield': 0.065, 'default_target': 28.50},
    '2223.SR': {'name': 'Luberef', 'moat': 'Base Oil Specialist', 'yield': 0.072, 'default_target': 130.00},
    '1050.SR': {'name': 'BSF', 'moat': 'Corporate Banking', 'yield': 0.045, 'default_target': 32.00},
    '7010.SR': {'name': 'stc', 'moat': 'Digital Backbone', 'yield': 0.052, 'default_target': 36.00},
    '4013.SR': {'name': 'HMG', 'moat': 'Healthcare Efficiency', 'yield': 0.019, 'default_target': 270.00},
    '1180.SR': {'name': 'SNB', 'moat': 'Giga-Project Capital', 'yield': 0.039, 'default_target': 34.00},
    '1111.SR': {'name': 'Tadawul', 'moat': 'Sole Operator', 'yield': 0.028, 'default_target': 180.00},
    '5110.SR': {'name': 'SEC', 'moat': 'National Grid', 'yield': 0.040, 'default_target': 19.50},
    '2083.SR': {'name': 'Marafiq', 'moat': 'Utility Monopoly', 'yield': 0.038, 'default_target': 62.00},
    '4030.SR': {'name': 'Bahri', 'moat': 'Maritime Logistics', 'yield': 0.048, 'default_target': 24.00},
    '4263.SR': {'name': 'SAL', 'moat': 'Air Cargo Monopoly', 'yield': 0.022, 'default_target': 230.00},
    '4031.SR': {'name': 'SGS', 'moat': 'Airport Operations', 'yield': 0.042, 'default_target': 33.00},
    '8313.SR': {'name': 'Rasan', 'moat': 'Network Effect', 'yield': 0.012, 'default_target': 40.00},
    '7217.SR': {'name': 'ELM', 'moat': 'Data Integration', 'yield': 0.015, 'default_target': 900.00},
    '2381.SR': {'name': 'Arabian Drilling', 'moat': 'Infrastructure', 'yield': 0.035, 'default_target': 140.00},
    '1211.SR': {'name': 'Maaden', 'moat': 'Resource Rights', 'yield': 0.000, 'default_target': 45.00},
    '2020.SR': {'name': 'SABIC AN', 'moat': 'Production Efficiency', 'yield': 0.044, 'default_target': 65.00},
    '2280.SR': {'name': 'Almarai', 'moat': 'Distribution Power', 'yield': 0.025, 'default_target': 52.00},
    '1830.SR': {'name': 'Leejam', 'moat': 'Fitness Scale', 'yield': 0.024, 'default_target': 190.00}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. إعدادات التيليجرام (باستخدام معرفاتك)
BOT_TOKEN = "8096609350:AAGDqysumgrjhlcOniM6D885j620QqCWpc8"
CHAT_ID = "7172975999"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try: requests.post(url, data=payload)
    except: pass

# 4. لوحة تحكم الأسعار المستهدفة (Sidebar) لجميع الشركات
st.sidebar.header("🎯 Target Price Controller")
custom_targets = {}
for t_id, info in moat_assets.items():
    name = info['name']
    default = info['default_target']
    custom_targets[name] = st.sidebar.slider(
        f"Target: {name}", 
        min_value=float(default * 0.5), 
        max_value=float(default * 1.5), 
        value=float(default),
        step=0.1
    )

# 5. محرك البيانات والتحليل
@st.cache_data
def load_data(symbols):
    data = yf.download(symbols + ['^TASI.SR'], start="2024-01-01", progress=False)['Close']
    data = data.ffill().dropna()
    return data.drop(columns=['^TASI.SR']).rename(columns=mapping), data['^TASI.SR']

assets_data, tasi_data = load_data(tickers)

# 6. عرض النتائج والتنبيهات
if not assets_data.empty:
    # حساب الأداء والتوزيغ
    mu = expected_returns.mean_historical_return(assets_data)
    S = risk_models.CovarianceShrinkage(assets_data).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.09))
    weights = ef.max_sharpe(risk_free_rate=0.04)
    clean_weights = ef.clean_weights()

    # قسم التنبيهات (Alerts) لجميع الشركات الـ 19
    st.markdown("---")
    st.subheader("Alerts")
    
    # عرض التنبيهات في شكل شبكة (Grid)
    rows = [list(custom_targets.keys())[i:i + 4] for i in range(0, len(custom_targets), 4)]
    for row in rows:
        cols = st.columns(4)
        for i, name in enumerate(row):
            price = assets_data[name].iloc[-1]
            target = custom_targets[name]
            with cols[i]:
                if price <= target:
                    st.error(f"🚨 BUY: {name}")
                    st.write(f"Price: {price:.2f}")
                    if st.button(f"Notify {name}", key=name):
                        send_telegram(f"*Archer Alert* 🎯\n\nAsset: {name}\nPrice: {price:.2f}\nTarget: {target:.2f}")
                        st.toast(f"Telegram sent for {name}")
                else:
                    st.success(f"✅ {name}")
                    st.caption(f"Price: {price:.2f} (Target: {target:.2f})")

    # جدول المحفظة والرسوم البيانية
    st.markdown("---")
    st.subheader("Final Optimized Portfolio")
    alloc_df = pd.DataFrame([{"Asset": n, "Weight": f"{w:.2%}", "Yield": f"{moat_assets[next(k for k,v in mapping.items() if v==n)]['yield']:.2%}"} 
                             for n, w in clean_weights.items() if w > 0])
    st.table(alloc_df)
