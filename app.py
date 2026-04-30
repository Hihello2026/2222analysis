import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# إعدادات الصفحة
st.set_page_config(page_title="Quantitative Equity Analysis", layout="wide")

# تصميم الواجهة
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

st.title("Strategic Equity Analysis: Income & Stability")
st.sidebar.header("Portfolio Configuration")

# الأصول المتبقية بعد حذف علم وتداول
tickers = ['7010.SR', '4007.SR', '2285.SR', '2083.SR']
mapping = {
    '7010.SR': 'stc',
    '4007.SR': 'Al Hammadi',
    '2285.SR': 'Arabian Mills',
    '2083.SR': 'Marafiq'
}

# مدخلات المستخدم
start_date = st.sidebar.date_input("Analysis Start Date", value=pd.to_datetime("2024-06-15"))
portfolio_value = st.sidebar.number_input("Total Portfolio Value (SAR)", min_value=1000, value=1000000)
risk_free_rate = 0.02

@st.cache_data
def get_portfolio_data(symbols, start):
    try:
        price_df = yf.download(symbols, start=start)['Close']
        if len(symbols) > 1:
            price_df.rename(columns=mapping, inplace=True)
        else:
            price_df = pd.DataFrame(price_df)
            price_df.columns = [mapping[symbols[0]]]
            
        div_info = {}
        for sym in symbols:
            ticker_obj = yf.Ticker(sym)
            y_val = ticker_obj.info.get('dividendYield')
            # تصحيح القياس
            if y_val:
                div_info[mapping[sym]] = float(y_val) if y_val < 1 else float(y_val) / 100
            else:
                div_info[mapping[sym]] = 0.0
        return price_df, div_info
    except Exception:
        return pd.DataFrame(), {}

price_data, dividend_yields = get_portfolio_data(tickers, start_date)

if not price_data.empty:
    # 1. شارت أداء الأسعار
    st.subheader("Asset Price Performance (SAR)")
    st.line_chart(price_data)

    try:
        # 2. النمذجة الرياضية
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0.05) # حد أدنى 5% لكل سهم
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate) 
        target_weights = ef.clean_weights()

        # 3. إدارة التوزيع والتحليل
        st.markdown("---")
        st.subheader("Portfolio Management & Yield Analysis")
        
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

        # 4. ملخص الأداء ونسبة شارب
        st.markdown("---")
        st.subheader("Institutional Performance Metrics")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Cap. Gain", f"{ret:.2%}")
        m2.metric("Portfolio Yield", f"{(total_income/portfolio_value):.2%}")
        m3.metric("Annual Volatility", f"{vol:.2%}")
        m4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"Optimization Error: {e}")
else:
    st.error("Data retrieval failed.")
