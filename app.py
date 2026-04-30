import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Institutional UI & Styling
st.set_page_config(page_title="Elite Equity Analysis", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; color: #1a1a1a; }
    [data-testid="stMetricValue"] { color: #0f172a; font-weight: 800; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 4px;
        border: 1px solid #e2e8f0;
    }
    div[data-testid="stTable"] { background-color: #ffffff; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

st.title("Strategic Asset Allocation")
st.markdown("Precision Portfolio Engineering: Growth & Dividends vs. TASI Benchmark")

# 2. Updated Portfolio (Replacing Al Rajhi with BSF)
moat_assets = {
    '2222.SR': {'name': 'Aramco', 'moat': 'Cost Leadership'},
    '2223.SR': {'name': 'Luberef', 'moat': 'Base Oil Specialist'},
    '2083.SR': {'name': 'Marafiq', 'moat': 'Utility Monopoly'},
    '5110.SR': {'name': 'SEC', 'moat': 'National Grid'},
    '1111.SR': {'name': 'Tadawul', 'moat': 'Sole Operator'},
    '1050.SR': {'name': 'BSF', 'moat': 'Corporate Banking Lead'}, # Added Saudi French Bank
    '1180.SR': {'name': 'SNB', 'moat': 'Giga-Project Capital'},
    '8313.SR': {'name': 'Rasan', 'moat': 'Network Effect'},
    '7217.SR': {'name': 'ELM', 'moat': 'Exclusive Data Integration'},
    '2381.SR': {'name': 'Arabian Drilling', 'moat': 'Critical Infrastructure'},
    '7010.SR': {'name': 'stc', 'moat': 'Digital Backbone'},
    '4030.SR': {'name': 'Bahri', 'moat': 'Maritime Logistics Lead'},
    '4263.SR': {'name': 'SAL', 'moat': 'Air Cargo Monopoly'},
    '4031.SR': {'name': 'SGS', 'moat': 'Airport Operations'},
    '4013.SR': {'name': 'HMG', 'moat': 'Premium Healthcare Efficiency'},
    '1211.SR': {'name': 'Maaden', 'moat': 'Resource Rights'},
    '2020.SR': {'name': 'SABIC AN', 'moat': 'Production Efficiency'},
    '2280.SR': {'name': 'Almarai', 'moat': 'Distribution Power'},
    '1830.SR': {'name': 'Leejam', 'moat': 'Fitness Scale'}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. Sidebar Configuration
st.sidebar.header("Backtest Configuration")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-06-15"))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
capital = st.sidebar.number_input("Total Capital (SAR)", value=1000000)

@st.cache_data
def get_strategic_data(symbols, start, end):
    try:
        data = yf.download(symbols + ['^TASI.SR'], start=start, end=end, progress=False)['Close']
        data = data.ffill().dropna(axis=1, thresh=len(data)*0.5).dropna()
        
        # Projections for 2026 Dividend Yields
        real_yields = {
            'Aramco': 0.065, 'stc': 0.052, 'Bahri': 0.048, 
            'BSF': 0.045, 'SNB': 0.039, 'Luberef': 0.072,
            'SEC': 0.040, 'HMG': 0.019, 'SABIC AN': 0.044,
            'Marafiq': 0.038, 'SGS': 0.042, 'Tadawul': 0.028,
            'Almarai': 0.025, 'SAL': 0.022, 'ELM': 0.015,
            'Rasan': 0.012, 'Maaden': 0.000, 'Leejam': 0.024,
            'Arabian Drilling': 0.035
        }
        
        div_yields = {mapping[s]: real_yields.get(mapping[s], 0.035) for s in symbols if s in data.columns}
        benchmark = data['^TASI.SR']
        assets = data.drop(columns=['^TASI.SR']).rename(columns=mapping)
        return assets, benchmark, div_yields
    except:
        return pd.DataFrame(), pd.Series(), {}

# 4. Engine Execution
if start_date < end_date:
    assets_data, tasi_data, div_yields = get_strategic_data(tickers, start_date, end_date)

    if not assets_data.empty:
        mu = expected_returns.mean_historical_return(assets_data)
        S = risk_models.CovarianceShrinkage(assets_data).ledoit_wolf()
        
        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.09))
        weights = ef.max_sharpe(risk_free_rate=0.04)
        clean_weights = ef.clean_weights()

        p_ret, p_vol, p_sharpe = ef.portfolio_performance(risk_free_rate=0.04)
        
        total_income = 0
        final_table = []
        for name, w in clean_weights.items():
            if w > 0:
                y = div_yields.get(name, 0.035)
                income = (w * capital) * y
                total_income += income
                logic = next((v['moat'] for k, v in moat_assets.items() if v['name'] == name), "N/A")
                final_table.append({
                    "Asset": name, "Moat Strategy": logic, "Weight": f"{w:.2%}",
                    "Div. Yield": f"{y:.2%}", "Est. Income (SAR)": f"{income:,.2f}"
                })

        # 5. Dashboard Output
        st.subheader("Performance & Dividend Intelligence")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Capital Return", f"{p_ret:.2%}")
        m2.metric("Portfolio Yield", f"{(total_income/capital):.2%}")
        m3.metric("Annual Div. Income", f"{total_income:,.2f} SAR")
        m4.metric("Sharpe Ratio", f"{p_sharpe:.2f}")

        st.markdown("---")
        st.subheader("Detailed Allocation (BSF Update)")
        st.table(pd.DataFrame(final_table))

        # Comparison Growth Chart
        st.subheader("Growth Comparison: Portfolio vs. TASI Index")
        portfolio_daily = assets_data.pct_change().dropna().dot(np.array([clean_weights.get(c, 0) for c in assets_data.columns]))
        st.line_chart(pd.DataFrame({
            'Portfolio': (1 + portfolio_daily).cumprod(),
            'TASI Market': (1 + tasi_data.pct_change().dropna()).cumprod()
        })) 
        # --- قسم نظام التنبيهات الذكي (Alert System) ---
st.markdown("---")
st.header("🎯 Archer Matrix Strategy Alerts")

# تحديد أسعار الشراء المستهدفة (مثال بناءً على تحليل القيمة)
target_prices = {
    'Aramco': 28.50,
    'stc': 36.00,
    'BSF': 32.00,
    'Luberef': 130.00,
    'HMG': 270.00
}

alerts_found = False
col_alerts = st.columns(len(target_prices))

for i, (ticker_name, t_price) in enumerate(target_prices.items()):
    # الحصول على السعر الحالي من البيانات التي تم سحبها سابقاً
    current_price = assets_data[ticker_name].iloc[-1]
    
    with col_alerts[i]:
        if current_price <= t_price:
            st.error(f"🚨 BUY ALERT: {ticker_name}")
            st.write(f"Current: {current_price:.2f}")
            st.write(f"Target: {t_price:.2f}")
            alerts_found = True
        else:
            st.success(f"✅ {ticker_name} Stable")
            st.caption(f"Price: {current_price:.2f}")

if not alerts_found:
    st.info("No immediate buy signals. All assets are currently above target entry prices.")
