import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Institutional Light Theme
st.set_page_config(page_title="Elite Dividend Analysis", layout="wide")

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
st.markdown("Dividend-Focused Portfolio Optimization vs. TASI")

# 2. Comprehensive Moat Portfolio
moat_assets = {
    '2222.SR': {'name': 'Aramco', 'moat': 'Cost Leadership'},
    '2223.SR': {'name': 'Luberef', 'moat': 'Base Oil Specialist'},
    '2083.SR': {'name': 'Marafiq', 'moat': 'Utility Monopoly'},
    '5110.SR': {'name': 'SEC', 'moat': 'National Grid'},
    '1111.SR': {'name': 'Tadawul', 'moat': 'Sole Operator'},
    '1120.SR': {'name': 'Al Rajhi', 'moat': 'Deposit Hegemony'},
    '1180.SR': {'name': 'SNB', 'moat': 'Giga-Project Capital'},
    '8313.SR': {'name': 'Rasan', 'moat': 'Network Effect'},
    '7217.SR': {'name': 'ELM', 'moat': 'Exclusive Data Integration'},
    '2381.SR': {'name': 'Arabian Drilling', 'moat': 'Critical Infrastructure'},
    '7010.SR': {'name': 'stc', 'moat': 'Digital Backbone'},
    '4030.SR': {'name': 'Bahri', 'moat': 'Maritime Logistics Lead'},
    '4263.SR': {'name': 'SAL', 'moat': 'Air Cargo Monopoly'},
    '4031.SR': {'name': 'SGS', 'moat': 'Airport Operations'},
    '4013.SR': {'name': 'HMG', 'moat': 'Premium Healthcare Efficiency'},
    '1211.SR': {'name': 'Maaden', 'moat': 'Resource Rights'},
    '2020.SR': {'name': 'SABIC AN', 'moat': 'Production Efficiency'},
    '2280.SR': {'name': 'Almarai', 'moat': 'Distribution Power'},
    '1830.SR': {'name': 'Leejam', 'moat': 'Fitness Scale'}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. Sidebar Configuration
st.sidebar.header("Backtest Configuration")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-06-15"))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
capital = st.sidebar.number_input("Capital (SAR)", value=1000000)

@st.cache_data
def get_dividend_data(symbols, start, end):
    try:
        # Get Prices and Dividends
        data = yf.download(symbols + ['^TASI.SR'], start=start, end=end, progress=False)['Close']
        data = data.ffill().dropna(axis=1, thresh=len(data)*0.5).dropna()
        
        div_yields = {}
        for s in symbols:
            if s in data.columns:
                ticker = yf.Ticker(s)
                # Fetching actual dividend yield from info
                y = ticker.info.get('dividendYield', 0.03)
                div_yields[mapping[s]] = float(y) if y and y < 1 else 0.03
                
        benchmark = data['^TASI.SR']
        assets = data.drop(columns=['^TASI.SR']).rename(columns=mapping)
        return assets, benchmark, div_yields
    except:
        return pd.DataFrame(), pd.Series(), {}

# 4. Processing and Display
if start_date < end_date:
    assets_data, tasi_data, div_yields = get_dividend_data(tickers, start_date, end_date)

    if not assets_data.empty:
        # Optimization Logic
        mu = expected_returns.mean_historical_return(assets_data)
        S = risk_models.CovarianceShrinkage(assets_data).ledoit_wolf()
        
        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.09))
        weights = ef.max_sharpe(risk_free_rate=0.04)
        clean_weights = ef.clean_weights()

        # Calculation Metrics
        p_ret, p_vol, p_sharpe = ef.portfolio_performance(risk_free_rate=0.04)
        
        # Portfolio Cumulative Dividend Calculation
        total_div_income = 0
        final_alloc = []
        for name, w in clean_weights.items():
            if w > 0:
                y = div_yields.get(name, 0.03)
                income = (w * capital) * y
                total_div_income += income
                final_alloc.append({
                    "Asset": name,
                    "Weight": f"{w:.2%}",
                    "Dividend Yield": f"{y:.2%}",
                    "Annual Income (SAR)": f"{income:,.2f}"
                })

        # Display Metrics
        st.subheader("Dividend & Performance Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Capital Gain", f"{p_ret:.2%}")
        c2.metric("Portfolio Yield", f"{(total_div_income/capital):.2%}")
        c3.metric("Annual Div. Income", f"{total_div_income:,.2f} SAR")
        c4.metric("Sharpe Ratio", f"{p_sharpe:.2f}")

        # Allocation Table
        st.markdown("---")
        st.subheader("Asset Allocation & Passive Income Breakdown")
        st.table(pd.DataFrame(final_alloc))

        # Comparison Chart
        st.markdown("---")
        st.subheader("Growth Chart: Portfolio vs. TASI")
        portfolio_daily = assets_data.pct_change().dropna().dot(np.array([clean_weights.get(c, 0) for c in assets_data.columns]))
        st.line_chart(pd.DataFrame({
            'Portfolio Growth': (1 + portfolio_daily).cumprod(),
            'TASI Growth': (1 + tasi_data.pct_change().dropna()).cumprod()
        }))
