import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np

# 1. Professional UI Setup
st.set_page_config(page_title="Strategic Equity Analysis", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; color: #1e293b; }
    [data-testid="stMetricValue"] { color: #0f172a; font-weight: 700; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    div[data-testid="stTable"] { background-color: #ffffff; border-radius: 8px; }
    h1, h2, h3 { color: #0f172a; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("Strategic Equity Analysis")
st.markdown("Advanced Asset Allocation for Sharpe Ratio Optimization")

# 2. Asset Dictionary
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
    '4263.SR': {'name': 'SAL', 'moat': 'Air Cargo Logistics Monopoly'},
    '4031.SR': {'name': 'SGS', 'moat': 'Airport Ground Operations'},
    '4007.SR': {'name': 'Al Hammadi', 'moat': 'Strategic Healthcare Delivery'},
    '1211.SR': {'name': 'Maaden', 'moat': 'Resource Scarcity Rights'},
    '2020.SR': {'name': 'SABIC AN', 'moat': 'Production Efficiency'},
    '2200.SR': {'name': 'Saudi Pipes', 'moat': 'Energy Supply Chain Niche'},
    '2280.SR': {'name': 'Almarai', 'moat': 'Cold-Chain Distribution Power'},
    '1830.SR': {'name': 'Leejam', 'moat': 'Fitness Network Scale'}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. Enhanced Parameters for High Sharpe
capital = st.sidebar.number_input("Total Capital (SAR)", value=1000000)
# Narrowing weights to enforce diversification (Essential for Sharpe > 1.0)
min_w = st.sidebar.slider("Min Weight (%)", 1, 3, 2) / 100
max_w = st.sidebar.slider("Max Weight (%)", 5, 12, 10) / 100 
risk_free = 0.04 

@st.cache_data
def get_market_data(symbols):
    try:
        data = yf.download(symbols, start="2024-06-15", progress=False)['Close']
        data = data.ffill().dropna(axis=1, thresh=len(data)*0.5).dropna()
        actual_symbols = [s for s in symbols if s in data.columns]
        data.rename(columns=mapping, inplace=True)
        
        div_yields = {}
        for s in actual_symbols:
            name = mapping[s]
            ticker_info = yf.Ticker(s).info
            y = ticker_info.get('dividendYield', 0)
            div_yields[name] = float(y) if y and y < 1 else (float(y)/100 if y else 0.035)
        return data, div_yields
    except:
        return pd.DataFrame(), {}

price_data, div_yields = get_market_data(tickers)

if not price_data.empty:
    st.subheader("Asset Price Performance")
    st.line_chart(price_data)

    try:
        # 4. Optimization Engine
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        # We target the Max Sharpe while strictly enforcing the 10% cap
        ef = EfficientFrontier(mu, S, weight_bounds=(min_w, max_w))
        
        # Optimization Target
        weights = ef.max_sharpe(risk_free_rate=risk_free)
        clean_weights = ef.clean_weights()

        st.markdown("---")
        st.subheader("Optimized Asset Allocation")
        
        allocation_data = []
        total_income = 0
        
        for name, weight in clean_weights.items():
            if weight > 0.001:
                y = div_yields.get(name, 0.035)
                income = (weight * capital) * y
                total_income += income
                moat_type = next((v['moat'] for k, v in moat_assets.items() if v['name'] == name), "N/A")
                
                allocation_data.append({
                    "Asset": name,
                    "Economic Moat": moat_type,
                    "Weight": f"{weight:.2%}",
                    "Yield": f"{y:.2%}",
                    "Est. Income (SAR)": f"{income:,.2f}"
                })

        st.table(pd.DataFrame(allocation_data))

        # 5. Institutional Metrics
        st.markdown("---")
        st.subheader("Portfolio Performance Summary")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected Return", f"{ret:.2%}")
        c2.metric("Portfolio Yield", f"{(total_income/capital):.2%}")
        c3.metric("Annual Volatility", f"{vol:.2%}")
        c4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"Mathematical Error: {e}")
