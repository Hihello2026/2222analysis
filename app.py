import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Quant Analysis - Institutional Grade", layout="wide")

st.title("🚀 Quantitative Analysis Platform - High-Growth Strategy")
st.sidebar.header("Advanced Portfolio Settings")

# Tickers: stc, Rasan, Arabian Mills, ELM, and Marafiq
# Combining Tech, Utility, Food, and Telecom for a "Mega-Quant" approach
tickers = ['7010.SR', '8313.SR', '2285.SR', '7203.SR', '2083.SR']

# Start Date Selector
start_date = st.sidebar.date_input("Analysis Start Date", value=pd.to_datetime("2024-06-15"))

# Data Fetching
@st.cache_data
def load_data(symbols, start):
    df = yf.download(symbols, start=start)['Close']
    return df

data = load_data(tickers, start_date)

if not data.empty:
    st.subheader("📈 Asset Universe Performance (stc, Rasan, Mills, ELM, Marafiq)")
    st.line_chart(data)

    try:
        # 1. Expected Returns (Annualized)
        mu = expected_returns.mean_historical_return(data)
        
        # 2. Risk Model (Covariance)
        S = risk_models.sample_cov(data)
        
        # 3. Efficient Frontier Construction
        ef = EfficientFrontier(mu, S)
        
        # 4. Optimization: Maximum Sharpe Ratio
        weights = ef.max_sharpe() 
        cleaned_weights = ef.clean_weights()

        # Display Resulting Weights
        st.subheader("⚖️ Optimized Portfolio Weights (Max Sharpe)")
        cols = st.columns(len(tickers))
        
        for i, ticker in enumerate(tickers):
            if ticker == '7010.SR': name = "stc"
            elif ticker == '8313.SR': name = "Rasan"
            elif ticker == '2285.SR': name = "Arabian Mills"
            elif ticker == '7203.SR': name = "ELM"
            else: name = "Marafiq"
            
            cols[i].metric(name, f"{cleaned_weights[ticker]:.2%}")

        # Performance Metrics
        ret, vol, sharpe = ef.portfolio_performance()
        
        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)
        
        # Return Status
        if ret >= 0.10:
            col_res1.success(f"**Expected Annual Return: {ret:.2%}** ✅")
        else:
            col_res1.info(f"**Expected Annual Return: {ret:.2%}**")
            
        col_res2.warning(f"**Annual Volatility: {vol:.2%}**")
        col_res3.success(f"**Sharpe Ratio: {sharpe:.2f}**")

    except Exception as e:
        st.error(f"Calculation Error: {e}")
        st.info("Ensure the start date aligns with all stocks' IPO dates.")
else:
    st.error("No data found.")

st.sidebar.markdown("""
---
**Manager's Commentary:**
By adding **Marafiq**, we introduce a defensive utility layer. The **Max Sharpe** objective will only allocate significant capital to Marafiq if its low-risk profile improves the overall efficiency of the high-growth Rasan/ELM core.
""")
