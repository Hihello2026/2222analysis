import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Quant Analysis - High Growth", layout="wide")

st.title("🚀 Quantitative Analysis Platform - High-Growth Strategy")
st.sidebar.header("Aggressive Portfolio Settings")

# Tickers: stc, Rasan, Arabian Mills, ELM
# ELM (7203.SR) and Rasan (8313.SR) are the primary growth drivers
tickers = ['7010.SR', '8313.SR', '2285.SR', '7203.SR']

# Start Date Selector
start_date = st.sidebar.date_input("Analysis Start Date", value=pd.to_datetime("2024-06-15"))

# Data Fetching
@st.cache_data
def load_data(symbols, start):
    df = yf.download(symbols, start=start)['Close']
    return df

data = load_data(tickers, start_date)

if not data.empty:
    st.subheader("📈 Performance of Growth Assets (stc, Rasan, Mills, ELM)")
    st.line_chart(data)

    try:
        # Advanced Quantitative Calculations
        # 1. Calculate Expected Annual Returns
        mu = expected_returns.mean_historical_return(data)
        
        # 2. Calculate Risk (Sample Covariance Matrix)
        S = risk_models.sample_cov(data)
        
        # 3. Build Efficient Frontier
        ef = EfficientFrontier(mu, S)
        
        # 4. Strategy: Maximize Sharpe Ratio (The 'Aggressive' Move)
        weights = ef.max_sharpe() 
        cleaned_weights = ef.clean_weights()

        # Display Optimal Allocation
        st.subheader("⚖️ Optimal Allocation for Maximum Efficiency (Max Sharpe)")
        cols = st.columns(len(tickers))
        
        for i, ticker in enumerate(tickers):
            if ticker == '7010.SR': name = "stc"
            elif ticker == '8313.SR': name = "Rasan"
            elif ticker == '2285.SR': name = "Arabian Mills"
            else: name = "ELM"
            
            cols[i].metric(name, f"{cleaned_weights[ticker]:.2%}")

        # Portfolio Performance Statistics
        ret, vol, sharpe = ef.portfolio_performance()
        
        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)
        
        # Highlight Return if > 10%
        if ret >= 0.10:
            col_res1.success(f"**Expected Annual Return: {ret:.2%}** ✅")
        else:
            col_res1.info(f"**Expected Annual Return: {ret:.2%}**")
            
        col_res2.warning(f"**Annual Volatility (Risk): {vol:.2%}**")
        col_res3.success(f"**Sharpe Ratio: {sharpe:.2f}**")

    except Exception as e:
        st.error(f"Calculation Error: {e}")
        st.info("Tip: Try extending the start date to provide more data points.")
else:
    st.error("No data found. Please check tickers or date range.")

st.sidebar.markdown("""
---
**Strategy Analysis:**
This version shifts from 'Risk Minimization' to 'Efficiency Maximization'. 
By optimizing the **Sharpe Ratio**, the model identifies the perfect balance between **Rasan's** momentum and **stc's** structural stability.
""")
