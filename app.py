import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Quantitative Equity Analysis", layout="wide")

# Custom CSS for Professional UI
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
        
        # Download dividend data with standardization
        div_info = {}
        for sym in symbols:
            ticker_obj = yf.Ticker(sym)
            y_val = ticker_obj.info.get('dividendYield')
            # Standardize yfinance yield (converts 0.04 to 4% or handles missing data)
            div_info[mapping[sym]] = float(y_val) if y_val is not None else 0.0
            
        return price_df, div_info
    except Exception:
        return pd.DataFrame(), {}

price_data, dividend_yields = get_portfolio_data(tickers, start_date)

if not price_data.empty:
    st.subheader("Asset Performance Visualization")
    st.line_chart(price_data)

    try:
        # 1. Quantitative Modeling
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        # 2. Optimization with 5% Floor (Minimum Allocation Constraint)
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

        # 4. Rebalancing & Yield Management
        st.markdown("---")
        st.subheader("Automated Management & Yield Analysis")
        
        mgmt_data = []
        total_income = 0
        
        for asset, t_weight in target_weights.items():
            # Handle drift calculation (handling potential NaN for new listings)
            current_p = price_data[asset].iloc[-1]
            initial_p = price_data[asset].iloc[0]
            perf = (current_p / initial_p) - 1 if initial_p > 0 else 0
            
            current_w = t_weight * (1 + perf)
            drift = current_w - t_weight
            
            # Standardized Dividend Logic
            y_rate = dividend_yields.get(asset, 0)
            asset_value = t_weight * portfolio_value
            annual_income = asset_value * y_rate
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

        # 5. Portfolio Performance Summary
        st.markdown("---")
        st.subheader("Institutional Portfolio Summary")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Cap. Gain", f"{ret:.2%}")
        m2.metric("Portfolio Yield", f"{(total_income/portfolio_value):.2%}")
        m3.metric("Annual Volatility", f"{vol:.2%}")
        m4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"Mathematical Optimization Error: {e}")
else:
    st.error("Data retrieval failed. Please check your network or ticker symbols.")

st.sidebar.markdown("""
---
**Technical Methodology**
Utilizing **Modern Portfolio Theory (MPT)** to maximize risk-adjusted returns (Sharpe Ratio). A **5% floor** is enforced to ensure structural diversification across all selected assets.
""")
