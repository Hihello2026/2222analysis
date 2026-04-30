import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Professional Minimalist UI
st.set_page_config(page_title="Elite Moat Analysis", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; color: #1a1a1a; }
    [data-testid="stMetricValue"] { color: #111827; font-weight: 800; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 4px;
        border: 1px solid #f1f5f9;
    }
    div[data-testid="stTable"] { background-color: #ffffff; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

st.title("Strategic Asset Allocation")
st.markdown("Advanced Moat Optimization vs. TASI Benchmark")

# 2. Refined Asset List (Replacing Hammadi with HMG)
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
    '4013.SR': {'name': 'HMG', 'moat': 'Premium Healthcare Efficiency'}, # Sulaiman Al Habib added
    '1211.SR': {'name': 'Maaden', 'moat': 'Resource Rights'},
    '2020.SR': {'name': 'SABIC AN', 'moat': 'Production Efficiency'},
    '2280.SR': {'name': 'Almarai', 'moat': 'Distribution Power'},
    '1830.SR': {'name': 'Leejam', 'moat': 'Fitness Scale'}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. Sidebar for Backtesting
st.sidebar.header("Backtest Settings")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-06-15"))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
capital = st.sidebar.number_input("Capital (SAR)", value=1000000)

@st.cache_data
def get_benchmarked_data(symbols, start, end):
    try:
        all_tickers = symbols + ['^TASI.SR']
        data = yf.download(all_tickers, start=start, end=end, progress=False)['Close']
        data = data.ffill().dropna(axis=1, thresh=len(data)*0.5).dropna()
        benchmark = data['^TASI.SR']
        assets = data.drop(columns=['^TASI.SR'])
        assets.rename(columns=mapping, inplace=True)
        return assets, benchmark
    except:
        return pd.DataFrame(), pd.Series()

# 4. Calculation and Comparison
if start_date < end_date:
    assets_data, tasi_data = get_benchmarked_data(tickers, start_date, end_date)

    if not assets_data.empty:
        # Optimization Strategy
        mu = expected_returns.mean_historical_return(assets_data)
        S = risk_models.CovarianceShrinkage(assets_data).ledoit_wolf() # More robust covariance estimation
        
        # Solving for Max Sharpe Ratio with 9% concentration cap
        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.09))
        weights = ef.max_sharpe(risk_free_rate=0.04)
        clean_weights = ef.clean_weights()

        # Visualizing Results
        portfolio_weights = np.array([clean_weights.get(col, 0) for col in assets_data.columns])
        portfolio_daily_ret = assets_data.pct_change().dropna().dot(portfolio_weights)
        portfolio_cum_ret = (1 + portfolio_daily_ret).cumprod()
        tasi_cum_ret = (1 + tasi_data.pct_change().dropna()).cumprod()

        st.subheader("Performance vs. TASI Benchmark")
        st.line_chart(pd.DataFrame({'Portfolio': portfolio_cum_ret, 'TASI Index': tasi_cum_ret}))

        # 5. Comparative Analytics
        p_ret, p_vol, p_sharpe = ef.portfolio_performance(risk_free_rate=0.04)
        
        tasi_daily_ret = tasi_data.pct_change().dropna()
        t_ann_ret = (tasi_cum_ret.iloc[-1] ** (252/len(tasi_data))) - 1
        t_ann_vol = tasi_daily_ret.std() * np.sqrt(252)
        t_sharpe = (t_ann_ret - 0.04) / t_ann_vol

        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Return", f"{p_ret:.2%}", f"{(p_ret - t_ann_ret):.2%}")
        c2.metric("Portfolio Volatility", f"{p_vol:.2%}", f"{(p_vol - t_ann_vol):.2%}", delta_color="inverse")
        c3.metric("Sharpe Ratio", f"{p_sharpe:.2f}", f"{(p_sharpe - t_sharpe):.2f}")

        # 6. Allocation Weights Table
        st.markdown("---")
        st.subheader("Asset Allocation & Moat Logic")
        alloc_table = []
        for name, w in clean_weights.items():
            if w > 0:
                logic = next((v['moat'] for key, v in moat_assets.items() if v['name'] == name), "")
                alloc_table.append({"Asset": name, "Moat Strategy": logic, "Weight": f"{w:.2%}"})
        st.table(pd.DataFrame(alloc_table))
