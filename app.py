import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Quantitative Equity Analysis", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e1e4e8;
    }
    .rebalance-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #fff4e6;
        border-left: 5px solid #ff922b;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Quantitative Equity Analysis Platform")
st.sidebar.header("Portfolio Configuration")

# Tickers and Mapping
tickers = ['7010.SR', '8313.SR', '2285.SR', '7203.SR', '2083.SR']
mapping = {
    '7010.SR': 'stc',
    '8313.SR': 'Rasan',
    '2285.SR': 'Arabian Mills',
    '7203.SR': 'ELM',
    '2083.SR': 'Marafiq'
}

# Sidebar Inputs
start_date = st.sidebar.date_input("Analysis Start Date", value=pd.to_datetime("2024-06-15"))
portfolio_value = st.sidebar.number_input("Total Portfolio Value (SAR)", min_value=1000, value=100000)
rebalance_threshold = st.sidebar.slider("Rebalancing Threshold (%)", 1, 10, 5) / 100

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
    # 1. Visualization & Correlation
    st.subheader("Asset Performance Visualization")
    st.line_chart(data)

    st.markdown("---")
    col_chart, col_corr = st.columns([1, 1])
    
    with col_corr:
        st.subheader("Asset Correlation Matrix")
        corr_matrix = data.pct_change().corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt='.2f', ax=ax, center=0)
        st.pyplot(fig)

    try:
        # 2. Optimization Logic
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe() 
        target_weights = ef.clean_weights()

        # 3. Automated Rebalancing Alerts Logic
        st.markdown("---")
        st.subheader("Automated Rebalancing Management")
        
        # Simulate Current Weights (in a real app, these would come from your brokerage API)
        # Here we assume a slight drift for demonstration
        st.write("System comparison between Current Market Weights and Model Targets.")
        
        rebalance_data = []
        for asset, t_weight in target_weights.items():
            current_price = data[asset].iloc[-1]
            initial_price = data[asset].iloc[0]
            # Simulating drift based on price action
            simulated_current_weight = t_weight * (current_price / initial_price) 
            drift = simulated_current_weight - t_weight
            
            status = "Optimal"
            if drift > rebalance_threshold: status = "Overweight (Sell)"
            elif drift < -rebalance_threshold: status = "Underweight (Buy)"
            
            rebalance_data.append({
                "Asset": asset,
                "Target Weight": f"{t_weight:.2%}",
                "Current Drift": f"{drift:+.2%}",
                "Status": status,
                "Action Amount (SAR)": f"{abs(drift * portfolio_value):,.2f}"
            })

        rebalance_df = pd.DataFrame(rebalance_data)
        
        # Highlight Alerts
        st.table(rebalance_df)

        # 4. Portfolio Analytics
        st.markdown("---")
        st.subheader("Portfolio Performance Metrics")
        ret, vol, sharpe = ef.portfolio_performance()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Annual Return", f"{ret:.2%}")
        m2.metric("Annual Volatility", f"{vol:.2%}")
        m3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"Optimization Error: {e}")
else:
    st.error("Data retrieval failed.")
