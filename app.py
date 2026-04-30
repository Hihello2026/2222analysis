import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# Page Configuration for a professional look
st.set_page_config(page_title="Quantitative Equity Analysis", layout="wide")

# Custom CSS to refine the UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e1e4e8;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Quantitative Equity Analysis Platform")
st.sidebar.header("Portfolio Configuration")

# Tickers: stc, Rasan, Arabian Mills, ELM, and Marafiq
tickers = ['7010.SR', '8313.SR', '2285.SR', '7203.SR', '2083.SR']

# Analysis Parameters
start_date = st.sidebar.date_input("Analysis Start Date", value=pd.to_datetime("2024-06-15"))
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0) / 100

# Data Acquisition
@st.cache_data
def load_data(symbols, start):
    try:
        df = yf.download(symbols, start=start)['Close']
        return df
    except Exception:
        return pd.DataFrame()

data = load_data(tickers, start_date)

if not data.empty:
    st.subheader("Asset Performance Visualization")
    st.line_chart(data)

    try:
        # Quantitative Modeling
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        # Optimization: Maximum Sharpe Ratio
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate) 
        cleaned_weights = ef.clean_weights()

        st.markdown("---")
        st.subheader("Optimal Portfolio Allocation")
        st.write("Calculated using the Maximum Sharpe Ratio objective to optimize risk-adjusted returns.")
        
        # Displaying weights in a clean grid
        cols = st.columns(len(tickers))
        for i, ticker in enumerate(tickers):
            mapping = {
                '7010.SR': 'stc',
                '8313.SR': 'Rasan',
                '2285.SR': 'Arabian Mills',
                '7203.SR': 'ELM',
                '2083.SR': 'Marafiq'
            }
            name = mapping.get(ticker, ticker)
            cols[i].metric(label=name, value=f"{cleaned_weights[ticker]:.2%}")

        # Performance Metrics Section
        st.markdown("---")
        st.subheader("Portfolio Analytics")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        m1, m2, m3 = st.columns(3)
        m1.write(f"**Expected Annual Return**")
        m1.title(f"{ret:.2%}")
        
        m2.write(f"**Annual Volatility**")
        m2.title(f"{vol:.2%}")
        
        m3.write(f"**Sharpe Ratio**")
        m3.title(f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"Mathematical Optimization Error: {e}")
else:
    st.error("Data retrieval failed. Verify connection and ticker symbols.")

st.sidebar.markdown("""
---
**Technical Methodology**
This platform utilizes the **Modern Portfolio Theory (MPT)** framework. By optimizing the Sharpe Ratio, the system identifies the tangency portfolio on the Efficient Frontier, maximizing the excess return per unit of volatility.
""")
