import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
def load_data(symbols, start):
    try:
        df = yf.download(symbols, start=start)['Close']
        df.rename(columns=mapping, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

data = load_data(tickers, start_date)

if not data.empty:
    st.subheader("Asset Performance Visualization")
    st.line_chart(data)

    st.markdown("---")
    st.subheader("Risk Management: Asset Correlation Matrix")
    corr_matrix = data.pct_change().corr()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt='.2f', ax=ax, center=0)
    st.pyplot(fig)

    try:
        # 1. Quantitative Modeling
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        # 2. Optimization with Constraints
        ef = EfficientFrontier(mu, S)
        
        # MANDATORY CONSTRAINT: Ensure at least 5% allocation per stock
        # This prevents the model from assigning 0% to "less efficient" assets
        ef.add_constraint(lambda w: w >= 0.05)
        
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate) 
        target_weights = ef.clean_weights()

        # 3. Optimal Allocation Display
        st.markdown("---")
        st.subheader("Optimal Portfolio Allocation")
        st.write("Allocation optimized for the Sharpe Ratio with a 5% minimum floor per asset to ensure sector diversification.")
        
        cols = st.columns(len(tickers))
        for i, ticker_name in enumerate(mapping.values()):
            cols[i].metric(label=ticker_name, value=f"{target_weights.get(ticker_name, 0):.2%}")

        # 4. Automated Rebalancing Alerts
        st.markdown("---")
        st.subheader("Automated Rebalancing Management")
        
        rebalance_list = []
        for asset, t_weight in target_weights.items():
            # Calculate drift based on recent performance
            performance = (data[asset].iloc[-1] / data[asset].iloc[0])
            current_weight = t_weight * performance
            drift = current_weight - t_weight
            
            status = "Optimal"
            if drift > rebalance_threshold: status = "Overweight (Sell)"
            elif drift < -rebalance_threshold: status = "Underweight (Buy)"
            
            rebalance_list.append({
                "Asset": asset,
                "Target %": f"{t_weight:.2%}",
                "Drift %": f"{drift:+.2%}",
                "Status": status,
                "Action Amount (SAR)": f"{abs(drift * portfolio_value):,.2f}"
            })

        st.table(pd.DataFrame(rebalance_list))

        # 5. Portfolio Analytics
        st.markdown("---")
        st.subheader("Portfolio Performance Summary")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Annual Return", f"{ret:.2%}")
        m2.metric("Annual Volatility", f"{vol:.2%}")
        m3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"Optimization Error: {e}")
else:
    st.error("Data retrieval failed. Please check tickers or network connection.")

st.sidebar.markdown("""
---
**Technical Methodology**
This platform employs **Modern Portfolio Theory (MPT)** with custom linear constraints. By enforcing a minimum 5% floor, the system mitigates **Concentration Risk** while optimizing for risk-adjusted returns via the **Tangency Portfolio**.
""")
