import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# إعدادات الصفحة
st.set_page_config(page_title="Quantitative Equity Analysis", layout="wide")

# تصميم واجهة مؤسسية
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

st.title("Strategic Equity Analysis: Growth vs. Yield")
st.sidebar.header("Portfolio Configuration")

# الأصول المختارة (تشمل الحمادي وتداول)
tickers = ['7010.SR', '4007.SR', '2285.SR', '7203.SR', '2083.SR', '1111.SR']
mapping = {
    '7010.SR': 'stc',
    '4007.SR': 'Al Hammadi',
    '2285.SR': 'Arabian Mills',
    '7203.SR': 'ELM',
    '2083.SR': 'Marafiq',
    '1111.SR': 'Tadawul'
}

# مدخلات المستخدم
start_date = st.sidebar.date_input("Analysis Start Date", value=pd.to_datetime("2024-06-15"))
portfolio_value = st.sidebar.number_input("Total Portfolio Value (SAR)", min_value=1000, value=1000000)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0) / 100

@st.cache_data
def get_portfolio_data(symbols, start):
    try:
        price_df = yf.download(symbols, start=start)['Close']
        price_df.rename(columns=mapping, inplace=True)
        
        div_info = {}
        for sym in symbols:
            ticker_obj = yf.Ticker(sym)
            y_val = ticker_obj.info.get('dividendYield')
            # تصحيح قياس التوزيعات
            if y_val:
                div_info[mapping[sym]] = float(y_val) if y_val < 1 else float(y_val) / 100
            else:
                div_info[mapping[sym]] = 0.0
            
        return price_df, div_info
    except Exception:
        return pd.DataFrame(), {}

price_data, dividend_yields = get_portfolio_data(tickers, start_date)

if not price_data.empty:
    # 1. مخطط أسعار الأصول (Price Chart)
    st.subheader("Asset Price Performance (SAR)")
    st.line_chart(price_data)

    try:
        # 2. النمذجة الرياضية (Optimization)
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        # تحسين المحفظة بناءً على نسبة شارب مع فرض حد أدنى 5% للتنويع
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0.05)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate) 
        target_weights = ef.clean_weights()

        # 3. إدارة التوزيع وعوائد الحمادي والأسهم الأخرى
        st.markdown("---")
        st.subheader("Strategic Management & Yield Analysis")
        
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

        # 4. ملخص مؤشرات الأداء (بما في ذلك نسبة شارب)
        st.markdown("---")
        st.subheader("Institutional Performance Metrics")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Cap. Gain", f"{ret:.2%}")
        m2.metric("Portfolio Yield", f"{(total_income/portfolio_value):.2%}")
        m3.metric("Annual Volatility", f"{vol:.2%}")
        m4.metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.sidebar.info(f"The Sharpe Ratio of {sharpe:.2f} indicates the risk-adjusted return relative to the risk-free rate.")

    except Exception as e:
        st.error(f"Optimization Error: {e}")
else:
    st.error("Data retrieval failed. Please verify ticker symbols.")
