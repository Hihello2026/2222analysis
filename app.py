import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import time

# إعدادات الصفحة الاحترافية
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
    </style>
    """, unsafe_allow_html=True)

st.title("Strategic Equity Analysis: Income & Stability")
st.sidebar.header("Portfolio Configuration")

# الأصول المختارة: stc، الحمادي، المطاحن العربية، المرافق
tickers = ['7010.SR', '4007.SR', '2285.SR', '2083.SR']
mapping = {
    '7010.SR': 'stc',
    '4007.SR': 'Al Hammadi',
    '2285.SR': 'Arabian Mills',
    '2083.SR': 'Marafiq'
}

# مدخلات المحفظة
start_date = st.sidebar.date_input("Analysis Start Date", value=pd.to_datetime("2024-06-15"))
portfolio_value = st.sidebar.number_input("Total Portfolio Value (SAR)", min_value=1000, value=1000000)
risk_free_rate = 0.02

@st.cache_data(show_spinner="جاري تحديث بيانات السوق...")
def get_portfolio_data(symbols, start):
    try:
        # محاولة جلب البيانات مع مهلة زمنية قصيرة
        price_df = yf.download(symbols, start=start, progress=False)['Close']
        
        if price_df.empty:
            return pd.DataFrame(), {}

        # إعادة تسمية الأعمدة
        if len(symbols) > 1:
            price_df.rename(columns=mapping, inplace=True)
        else:
            price_df = pd.DataFrame(price_df)
            price_df.columns = [mapping[symbols[0]]]
            
        div_info = {}
        for sym in symbols:
            ticker_obj = yf.Ticker(sym)
            # جلب عائد التوزيعات مع معالجة الأخطاء
            try:
                y_val = ticker_obj.info.get('dividendYield')
                if y_val:
                    div_info[mapping[sym]] = float(y_val) if y_val < 1 else float(y_val) / 100
                else:
                    div_info[mapping[sym]] = 0.0
            except:
                div_info[mapping[sym]] = 0.0
                
        return price_df, div_info
    except Exception as e:
        st.sidebar.error(f"Connection Error: {e}")
        return pd.DataFrame(), {}

# جلب البيانات
price_data, dividend_yields = get_portfolio_data(tickers, start_date)

if not price_data.empty:
    st.subheader("Asset Price Performance (SAR)")
    st.line_chart(price_data)

    try:
        # الحسابات الكمية للمحفظة
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        ef = EfficientFrontier(mu, S)
        # فرض حد أدنى للتنويع (5%)
        ef.add_constraint(lambda w: w >= 0.05)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate) 
        target_weights = ef.clean_weights()

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

        st.markdown("---")
        st.subheader("Institutional Performance Metrics")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Cap. Gain", f"{ret:.2%}")
        m2.metric("Portfolio Yield", f"{(total_income/portfolio_value):.2%}")
        m3.metric("Annual Volatility", f"{vol:.2%}")
        m4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"Mathematical Error: {e}")
else:
    st.warning("⚠️ لم يتم العثور على بيانات. يرجى التأكد من اتصال الإنترنت أو المحاولة مرة أخرى لاحقاً.")
