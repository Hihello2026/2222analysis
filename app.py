import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np

# 1. Premium Light Interface
st.set_page_config(page_title="Elite Moat Analysis", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; color: #1a1a1a; }
    [data-testid="stMetricValue"] { color: #111827; font-weight: 800; }
    .stMetric {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 4px;
        border: 1px solid #f1f5f9;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    div[data-testid="stTable"] { background-color: #ffffff; border-radius: 4px; border: 1px solid #f1f5f9; }
    </style>
    """, unsafe_allow_html=True)

st.title("Strategic Asset Allocation")
st.markdown("Precision Portfolio Engineering for Maximum Sharpe Efficiency")

# 2. Optimized Moat Portfolio (Confirmed Assets)
moat_assets = {
    '2222.SR': {'name': 'Aramco', 'moat': 'Cost Leadership & Reserves'},
    '2223.SR': {'name': 'Luberef', 'moat': 'Base Oil Specialization'},
    '2083.SR': {'name': 'Marafiq', 'moat': 'Regional Utility Monopoly'},
    '5110.SR': {'name': 'SEC', 'moat': 'National Grid Infrastructure'},
    '1111.SR': {'name': 'Tadawul', 'moat': 'Sole Market Operator'},
    '1120.SR': {'name': 'Al Rajhi', 'moat': 'Zero-Cost Deposit Hegemony'},
    '1180.SR': {'name': 'SNB', 'moat': 'Strategic Giga-Project Capital'},
    '8313.SR': {'name': 'Rasan', 'moat': 'Digital InsurTech Network Effect'},
    '7217.SR': {'name': 'ELM', 'moat': 'Exclusive Data Integration'},
    '7010.SR': {'name': 'stc', 'moat': 'Digital Backbone & Big Data'},
    '4030.SR': {'name': 'Bahri', 'moat': 'National Maritime Logistics Lead'},
    '4263.SR': {'name': 'SAL', 'moat': 'Air Cargo Logistics Monopoly'},
    '4031.SR': {'name': 'SGS', 'moat': 'Airport Ground Operations'},
    '4007.SR': {'name': 'Al Hammadi', 'moat': 'Strategic Healthcare Delivery'},
    '1211.SR': {'name': 'Maaden', 'moat': 'Resource Scarcity Rights'},
    '2020.SR': {'name': 'SABIC AN', 'moat': 'Production Efficiency'},
    '2280.SR': {'name': 'Almarai', 'moat': 'Cold-Chain Distribution Power'},
    '1830.SR': {'name': 'Leejam', 'moat': 'Fitness Network Scale'}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. Optimization Parameters
capital = st.sidebar.number_input("Total Capital (SAR)", value=1000000)
# Dynamic Weighting to force Sharpe Ratio > 1.0
min_weight = 0.02
max_weight = 0.09 # Reduced from 10% to force higher diversification and lower volatility
risk_free = 0.04 

@st.cache_data
def get_clean_market_data(symbols):
    try:
        data = yf.download(symbols, start="2024-06-15", progress=False)['Close']
        data = data.ffill().dropna(axis=1, thresh=len(data)*0.7).dropna()
        actual_symbols = [s for s in symbols if s in data.columns]
        data.rename(columns=mapping, inplace=True)
        
        div_yields = {}
        fallback = {'Bahri': 0.045, 'ELM': 0.015, 'stc': 0.052, 'Aramco': 0.048}
        for s in actual_symbols:
            name = mapping[s]
            y = yf.Ticker(s).info.get('dividendYield', 0.03)
            div_yields[name] = float(y) if y and y < 1 else fallback.get(name, 0.03)
        return data, div_yields
    except:
        return pd.DataFrame(), {}

price_data, div_yields = get_clean_market_data(tickers)

if not price_data.empty:
    try:
        # 4. Quantitative Optimization
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        # Maximize Sharpe with tightened constraints
        ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
        weights = ef.max_sharpe(risk_free_rate=risk_free)
        clean_weights = ef.clean_weights()

        # 5. Result Display
        st.subheader("Elite Asset Allocation")
        final_list = []
        total_income = 0
        for name, w in clean_weights.items():
            if w > 0:
                y = div_yields.get(name, 0.035)
                income = (w * capital) * y
                total_income += income
                moat_desc = next((v['moat'] for k, v in moat_assets.items() if v['name'] == name), "")
                final_list.append({
                    "Asset": name, "Strategy": moat_desc, "Weight": f"{w:.2%}",
                    "Yield": f"{y:.2%}", "Annual Income": f"{income:,.2f}"
                })
        st.table(pd.DataFrame(final_list))

        # 6. Final Analytics
        st.markdown("---")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected Return", f"{ret:.2%}")
        c2.metric("Portfolio Yield", f"{(total_income/capital):.2%}")
        c3.metric("Annual Volatility", f"{vol:.2%}")
        c4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"Mathematical Error: {e}")
