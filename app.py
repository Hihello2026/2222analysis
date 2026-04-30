import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Premium Light Interface
st.set_page_config(page_title="Institutional Backtesting", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; color: #1a1a1a; }
    [data-testid="stMetricValue"] { color: #111827; font-weight: 800; }
    .stMetric {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 4px;
        border: 1px solid #f1f5f9;
    }
    div[data-testid="stTable"] { background-color: #ffffff; border-radius: 4px; border: 1px solid #f1f5f9; }
    </style>
    """, unsafe_allow_html=True)

st.title("Strategic Asset Allocation")
st.markdown("Dynamic Portfolio Backtesting and Optimization")

# 2. Optimized Moat Portfolio
moat_assets = {
    '2222.SR': {'name': 'Aramco', 'moat': 'Cost Leadership & Reserves'},
    '2223.SR': {'name': 'Luberef', 'moat': 'Base Oil Specialization'},
    '2083.SR': {'name': 'Marafiq', 'moat': 'Regional Utility Monopoly'},
    '5110.SR': {'name': 'SEC', 'moat': 'National Grid Infrastructure'},
    '1111.SR': {'name': 'Tadawul', 'moat': 'Sole Market Operator'},
    '1120.SR': {'name': 'Al Rajhi', 'moat': 'Zero-Cost Deposit Hegemony'},
    '1180.SR': {'name': 'SNB', 'moat': 'Strategic Giga-Project Capital'},
    '8313.SR': {'name': 'Rasan', 'moat': 'Digital InsurTech Network Effect'},
    '7217.SR': {'name': 'ELM', 'moat': 'Exclusive Data Integration'},
    '7010.SR': {'name': 'stc', 'moat': 'Digital Backbone & Big Data'},
    '4030.SR': {'name': 'Bahri', 'moat': 'National Maritime Logistics Lead'},
    '4263.SR': {'name': 'SAL', 'moat': 'Air Cargo Logistics Monopoly'},
    '4031.SR': {'name': 'SGS', 'moat': 'Airport Ground Operations'},
    '4007.SR': {'name': 'Al Hammadi', 'moat': 'Strategic Healthcare Delivery'},
    '1211.SR': {'name': 'Maaden', 'moat': 'Resource Scarcity Rights'},
    '2020.SR': {'name': 'SABIC AN', 'moat': 'Production Efficiency'},
    '2280.SR': {'name': 'Almarai', 'moat': 'Cold-Chain Distribution Power'},
    '1830.SR': {'name': 'Leejam', 'moat': 'Fitness Network Scale'}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. Sidebar Configuration (Backtesting Dates Added)
st.sidebar.header("Backtest Configuration")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-06-15"))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

capital = st.sidebar.number_input("Total Capital (SAR)", value=1000000)
max_weight = 0.09 
risk_free = 0.04 

@st.cache_data
def get_backtest_data(symbols, start, end):
    try:
        # Fetching data for the specific backtest period
        data = yf.download(symbols, start=start, end=end, progress=False)['Close']
        data = data.ffill().dropna(axis=1, thresh=len(data)*0.5).dropna()
        actual_symbols = [s for s in symbols if s in data.columns]
        data.rename(columns=mapping, inplace=True)
        
        div_yields = {}
        for s in actual_symbols:
            name = mapping[s]
            y = yf.Ticker(s).info.get('dividendYield', 0.035)
            div_yields[name] = float(y) if y and y < 1 else 0.035
        return data, div_yields
    except:
        return pd.DataFrame(), {}

# 4. Execute Backtest
if start_date < end_date:
    price_data, div_yields = get_backtest_data(tickers, start_date, end_date)

    if not price_data.empty:
        st.subheader(f"Historical Performance ({start_date} to {end_date})")
        st.line_chart(price_data)

        try:
            # Quantitative Optimization for the selected period
            mu = expected_returns.mean_historical_return(price_data)
            S = risk_models.sample_cov(price_data)
            
            ef = EfficientFrontier(mu, S, weight_bounds=(0.02, max_weight))
            weights = ef.max_sharpe(risk_free_rate=risk_free)
            clean_weights = ef.clean_weights()

            # 5. Result Display
            st.subheader("Elite Asset Allocation (Backtested)")
            final_list = []
            total_income = 0
            for name, w in clean_weights.items():
                if w > 0:
                    y = div_yields.get(name, 0.035)
                    income = (w * capital) * y
                    total_income += income
                    final_list.append({
                        "Asset": name, "Weight": f"{w:.2%}",
                        "Historical Yield": f"{y:.2%}", "Annual Income": f"{income:,.2f}"
                    })
            st.table(pd.DataFrame(final_list))

            # 6. Final Analytics
            st.markdown("---")
            ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Expected Return", f"{ret:.2%}")
            c2.metric("Portfolio Yield", f"{(total_income/capital):.2%}")
            c3.metric("Annual Volatility", f"{vol:.2%}")
            c4.metric("Sharpe Ratio", f"{sharpe:.2f}")

        except Exception as e:
            st.error(f"Backtest Error: {e}")
else:
    st.error("Error: Start Date must be before End Date.")
