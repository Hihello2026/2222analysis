import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# Page Configuration for a professional institutional look
st.set_page_config(page_title="Quantitative Equity Analysis", layout="wide")

# Custom CSS for a clean, banking-style UI
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

# Tickers and Professional Mapping
tickers = ['7010.SR', '8313.SR', '2285.SR', '7203.SR', '2083.SR', '1111.SR']
mapping = {
    '7010.SR': 'stc',
    '8313.SR': 'Rasan',
    '2285.SR': 'Arabian Mills',
    '7203.SR': 'ELM',
    '2083.SR': 'Marafiq',
    '1111.SR': 'Tadawul'
}

# Sidebar Inputs
start_date = st.sidebar.date_input("Analysis Start Date", value=pd.to_datetime("2024-06-15"))
portfolio_value = st.sidebar.number_input("Total Portfolio Value (SAR)", min_value=1000, value=100000)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0) / 100
rebalance_threshold = st.sidebar.slider("Drift Threshold (%)", 1, 10, 5) / 100

@st.cache_data
def get_portfolio_data(symbols, start):
    try:
        # Download price data
        price_df = yf.download(symbols, start=start)['Close']
        price_df.rename(columns=mapping, inplace=True)
        
        # Download and standardize dividend data
        div_info = {}
        for sym in symbols:
            ticker_obj = yf.Ticker(sym)
            y_val = ticker_obj.info.get('dividendYield')
            # Fix: Ensure yield is treated as a decimal (e.g., 0.05 for 5%)
            # If yfinance returns 5.0, we divide by 100. If 0.05, we keep it.
            if y_val:
                div_info[mapping[sym]] = float(y_val) if y_val < 1 else float(y_val) / 100
            else:
                div_info[mapping[sym]] = 0.0
            
        return price_df, div_info
    except Exception:
        return pd.DataFrame(), {}

price_data, dividend_yields = get_portfolio_data(tickers, start_date)

if not price_data.empty:
    st.subheader("Asset Performance Visualization")
    st.line_chart(price_data)

    try:
        # 1. Quantitative Modeling (Mean-Variance Optimization)
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        # 2. Optimization with 5% Minimum Floor per Asset
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0.05)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate) 
        target_weights = ef.clean_weights()

        # 3. Optimal Allocation Dashboard
        st.markdown("---")
        st.subheader("Optimal Portfolio Allocation")
        cols = st.columns(len(tickers))
        for i, ticker_name in enumerate(mapping.values()):
            cols[i].metric(label=ticker_name, value=f"{target_weights.get(ticker_name, 0):.2%}")

        # 4. Automated Management & Corrected Yield Analysis
        st.markdown("---")
        st.subheader("Automated Management & Yield Analysis")
        
        mgmt_data = []
        total_income = 0
        
        for asset, t_weight in target_weights.items():
            # Current value based on performance
            current_p = price_data[asset].iloc[-1]
            initial_p = price_data[asset].iloc[0]
            perf = (current_p / initial_p) - 1 if initial_p > 0 else 0
            
            current_w = t_weight * (1 + perf)
            drift = current_w - t_weight
            
            # Corrected Dividend Calculation
            y_rate = dividend_yields.get(asset, 0)
            asset_sar_value = t_weight * portfolio_value
            annual_income = asset_sar_value * y_rate
            total_income += annual_income
            
            status = "Optimal"
            if drift > rebalance_threshold: status = "Overweight (Sell)"
            elif drift < -rebalance_threshold: status = "Underweight (Buy)"
            
            mgmt_data.append({
                "Asset": asset,
                "Target %": f"{t_weight:.2%}",
                "Drift %": f"{drift:+.2%}",
                "Div. Yield": f"{y_rate:.2%}",
                "Est. Annual Income (SAR)": f"{annual_income:,.2f}",
                "Action": status
            })

        st.table(pd.DataFrame(mgmt_data))

        # 5. Institutional Portfolio Summary
        st.markdown("---")
        st.subheader("Portfolio Performance Summary")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Cap. Gain", f"{ret:.2%}")
        m2.metric("Portfolio Yield", f"{(total_income/portfolio_value):.2%}")
        m3.metric("Annual Volatility", f"{vol:.2%}")
        m4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"Optimization Error: {e}")
else:
    st.error("Data retrieval failed. Please check your connection.")
