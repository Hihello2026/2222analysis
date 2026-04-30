import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Quantitative Equity Analysis", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
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

# Tickers
tickers = ['7010.SR', '8313.SR', '2285.SR', '7203.SR', '2083.SR']
mapping = {
    '7010.SR': 'stc',
    '8313.SR': 'Rasan',
    '2285.SR': 'Arabian Mills',
    '7203.SR': 'ELM',
    '2083.SR': 'Marafiq'
}

start_date = st.sidebar.date_input("Analysis Start Date", value=pd.to_datetime("2024-06-15"))
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0) / 100

@st.cache_data
def load_data(symbols, start):
    try:
        df = yf.download(symbols, start=start)['Close']
        df.rename(columns=mapping, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

data = load_data(tickers, start_date)

if not data.empty:
    # 1. Asset Performance Chart
    st.subheader("Asset Performance Visualization")
    st.line_chart(data)

    # 2. Correlation Matrix Section
    st.markdown("---")
    st.subheader("Risk Management: Asset Correlation Matrix")
    st.write("This heatmap illustrates the statistical relationship between assets. Lower correlation indicates superior diversification benefits.")
    
    corr_matrix = data.pct_change().corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt='.2f', ax=ax, center=0)
    plt.title("Asset Correlation Heatmap")
    st.pyplot(fig)

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
        
        # Displaying weights
        cols = st.columns(len(tickers))
        for i, ticker_name in enumerate(mapping.values()):
            cols[i].metric(label=ticker_name, value=f"{cleaned_weights.get(ticker_name, 0):.2%}")

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
    st.error("Data retrieval failed.")

st.sidebar.markdown("""
---
**Technical Methodology**
By incorporating the **Correlation Matrix**, the model assesses systemic risk. Lower asset correlation reduces total portfolio variance without diminishing expected returns.
""")
