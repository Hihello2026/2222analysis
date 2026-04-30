import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np

# 1. Page Configuration & Professional Styling
st.set_page_config(page_title="Saudi Moat Portfolio", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for an Attractive "Modern Industrial" Interface
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stTable"] {
        background-color: #1f2937;
        border-radius: 10px;
        overflow: hidden;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 5px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Saudi Economic Moat Portfolio")
st.markdown("### *Institutional Asset Allocation for 2026*")

# 2. Moat Assets Dictionary (Refined for Growth & Stability)
moat_assets = {
    '2222.SR': {'name': 'Aramco', 'moat': 'Cost Leadership & Reserves'},
    '2223.SR': {'name': 'Luberef', 'moat': 'Niche Base Oil Specialist'},
    '2083.SR': {'name': 'Marafiq', 'moat': 'Natural Monopoly (Industrial Cities)'},
    '5110.SR': {'name': 'SEC', 'moat': 'National Grid Monopoly'},
    '1111.SR': {'name': 'Tadawul', 'moat': 'Regulatory Moat (Sole Operator)'},
    '1120.SR': {'name': 'Al Rajhi', 'moat': 'Retail Dominance & Zero-Cost Deposits'},
    '1180.SR': {'name': 'SNB', 'moat': 'Giga-Project Financing Leader'},
    '8313.SR': {'name': 'Rasan', 'moat': 'Network Effect (Tameeni Platform)'},
    '7217.SR': {'name': 'ELM', 'moat': 'Exclusive Data/Government Integration'},
    '7010.SR': {'name': 'stc', 'moat': 'Infrastructure & Big Data'},
    '4263.SR': {'name': 'SAL', 'moat': 'Air Cargo Handling Hegemony'},
    '4031.SR': {'name': 'SGS', 'moat': 'Airport Ground Ops Backbone'},
    '4007.SR': {'name': 'Al Hammadi', 'moat': 'Operational Efficiency (Riyadh)'},
    '1211.SR': {'name': 'Maaden', 'moat': 'Resource Scarcity & Mining Rights'},
    '2020.SR': {'name': 'SABIC AN', 'moat': 'Global Production Efficiency'},
    '2200.SR': {'name': 'Saudi Pipes', 'moat': 'Strategic Energy Supplier'},
    '2280.SR': {'name': 'Almarai', 'moat': 'Logistical Distribution Power'},
    '1830.SR': {'name': 'Leejam', 'moat': 'Fitness Sector Dominance'}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. Sidebar Configuration
st.sidebar.header("🕹️ Portfolio Controls")
capital = st.sidebar.number_input("Total Capital (SAR)", value=1000000, step=100000)
min_weight = st.sidebar.slider("Min Weight per Stock (%)", 1, 5, 2) / 100
max_weight = st.sidebar.slider("Max Weight per Stock (%)", 5, 15, 12) / 100 # Capped at 15% to boost Sharpe
risk_free = 0.035 # Updated for 2026 rates

@st.cache_data
def get_market_data(symbols):
    try:
        data = yf.download(symbols, start="2024-06-15", progress=False)['Close']
        data = data.ffill().dropna(axis=1, thresh=len(data)*0.5).dropna()
        actual_symbols = [s for s in symbols if s in data.columns]
        data.rename(columns=mapping, inplace=True)
        
        # Enhanced Dividend Yield Handling
        div_yields = {}
        fallback = {'stc': 0.052, 'Aramco': 0.048, 'SEC': 0.038, 'ELM': 0.015, 'Rasan': 0.01}
        for s in actual_symbols:
            name = mapping[s]
            info = yf.Ticker(s).info
            y = info.get('dividendYield', 0)
            div_yields[name] = float(y) if y and y < 1 else (float(y)/100 if y else fallback.get(name, 0.03))
        return data, div_yields
    except:
        return pd.DataFrame(), {}

price_data, div_yields = get_market_data(tickers)

if not price_data.empty:
    st.subheader("📈 Performance Benchmarking")
    st.line_chart(price_data)

    try:
        # 4. Quantitative Optimization (Sharpe Optimization)
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        # Solving for Maximum Sharpe Ratio
        ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
        weights = ef.max_sharpe(risk_free_rate=risk_free)
        clean_weights = ef.clean_weights()

        # 5. Dashboard Display
        st.markdown("---")
        st.subheader("📋 Targeted Asset Allocation")
        
        display_list = []
        total_div_income = 0
        
        for name, weight in clean_weights.items():
            if weight > 0.001:
                y = div_yields.get(name, 0.03)
                income = (weight * capital) * y
                total_div_income += income
                moat_desc = next((v['moat'] for k, v in moat_assets.items() if v['name'] == name), "Economic Moat")
                
                display_list.append({
                    "Asset": name,
                    "Moat Logic": moat_desc,
                    "Allocation": f"{weight:.2%}",
                    "Div. Yield": f"{y:.2%}",
                    "Est. Income (SAR)": f"{income:,.2f}"
                })

        st.table(pd.DataFrame(display_list))

        # 6. Portfolio Analytics (Focusing on Sharpe > 1.0)
        st.markdown("---")
        st.subheader("📊 Institutional Analytics")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected Growth", f"{ret:.2%}")
        c2.metric("Portfolio Yield", f"{(total_div_income/capital):.2%}")
        c3.metric("Annual Volatility", f"{vol:.2%}")
        c4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"Optimization Failure: {e}")
