import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# Institutional UI Styling
st.set_page_config(page_title="Quantitative Equity Analysis", layout="wide")
st.title("Quantitative Equity Analysis Platform")

# Updated Asset Universe (Rasan Removed)
tickers = ['7010.SR', '2285.SR', '7203.SR', '2083.SR', '1111.SR']
mapping = {'7010.SR': 'stc', '2285.SR': 'Arabian Mills', '7203.SR': 'ELM', '2083.SR': 'Marafiq', '1111.SR': 'Tadawul'}

portfolio_value = st.sidebar.number_input("Total Portfolio Value (SAR)", min_value=1000, value=1000000)
risk_free_rate = 0.02

@st.cache_data
def get_clean_data(symbols):
    df = yf.download(symbols, start="2024-06-15")['Close']
    df.rename(columns=mapping, inplace=True)
    
    div_info = {}
    for sym in symbols:
        ticker_obj = yf.Ticker(sym)
        y_val = ticker_obj.info.get('dividendYield', 0)
        # SAFETY CHECK: If yield is > 1 (e.g. 5.16), divide by 100 to get 0.0516
        if y_val is not None:
            div_info[mapping[sym]] = float(y_val) if y_val < 1 else float(y_val) / 100
        else:
            div_info[mapping[sym]] = 0.0
    return df, div_info

price_data, dividend_yields = get_clean_data(tickers)

if not price_data.empty:
    mu = expected_returns.mean_historical_return(price_data)
    S = risk_models.sample_cov(price_data)
    ef = EfficientFrontier(mu, S)
    ef.add_constraint(lambda w: w >= 0.05) # Maintain 5% floor
    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    target_weights = ef.clean_weights()

    st.subheader("Automated Management & Yield Analysis")
    mgmt_data = []
    total_income = 0
    
    for asset, t_weight in target_weights.items():
        y_rate = dividend_yields.get(asset, 0)
        annual_income = (t_weight * portfolio_value) * y_rate
        total_income += annual_income
        
        mgmt_data.append({
            "Asset": asset,
            "Target %": f"{t_weight:.2%}",
            "Div. Yield": f"{y_rate:.2%}",
            "Est. Annual Income (SAR)": f"{annual_income:,.2f}",
            "Action": "Optimal"
        })
    st.table(pd.DataFrame(mgmt_data))

    st.subheader("Portfolio Performance Summary")
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected Cap. Gain", f"{ret:.2%}")
    m2.metric("Portfolio Yield", f"{(total_income/portfolio_value):.2%}")
    m3.metric("Annual Volatility", f"{vol:.2%}")
    m4.metric("Sharpe Ratio", f"{sharpe:.2f}")
