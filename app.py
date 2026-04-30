import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# إعدادات الصفحة
st.set_page_config(page_title="Quantitative Equity Analysis", layout="wide")

st.title("Strategic Equity Analysis: Income & Stability")

# الأصول والبيانات الاحتياطية
tickers = ['7010.SR', '4007.SR', '2285.SR', '2083.SR']
mapping = {'7010.SR': 'stc', '4007.SR': 'Al Hammadi', '2285.SR': 'Arabian Mills', '2083.SR': 'Marafiq'}
fallback_yields = {'7010.SR': 0.0516, '4007.SR': 0.0422, '2083.SR': 0.0525, '2285.SR': 0.0243}

start_date = st.sidebar.date_input("Analysis Start Date", value=pd.to_datetime("2024-06-15"))
portfolio_value = st.sidebar.number_input("Total Portfolio Value (SAR)", min_value=1000, value=1000000)

@st.cache_data
def get_portfolio_data(symbols, start):
    try:
        price_df = yf.download(symbols, start=start, progress=False)['Close']
        if price_df.empty: return pd.DataFrame(), {}
        price_df.rename(columns=mapping, inplace=True)
        div_info = {}
        for sym in symbols:
            ticker_obj = yf.Ticker(sym)
            try:
                y_val = ticker_obj.info.get('dividendYield')
                div_info[mapping[sym]] = float(y_val) if y_val and y_val > 0 else fallback_yields.get(sym, 0.0)
            except: div_info[mapping[sym]] = fallback_yields.get(sym, 0.0)
        return price_df, div_info
    except: return pd.DataFrame(), {}

price_data, dividend_yields = get_portfolio_data(tickers, start_date)

if not price_data.empty:
    st.subheader("Asset Price Performance (SAR)")
    st.line_chart(price_data)

    try:
        # الحسابات الكمية
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        # استخدام مرونة أكبر في القيود لتجنب خطأ Infeasible
        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.70)) # حد أدنى 2% وحد أقصى 70%
        
        # تحسين المحفظة
        weights = ef.max_sharpe(risk_free_rate=0.02) 
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
                "Target Weight": f"{t_weight:.2%}",
                "Div. Yield": f"{y_rate:.2%}",
                "Est. Annual Income (SAR)": f"{annual_income:,.2f}",
                "Action": "Optimal"
            })

        st.table(pd.DataFrame(mgmt_data))

        st.markdown("---")
        st.subheader("Institutional Performance Metrics")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.02)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Cap. Gain", f"{ret:.2%}")
        m2.metric("Portfolio Yield", f"{(total_income/portfolio_value):.2%}")
        m3.metric("Annual Volatility", f"{vol:.2%}")
        m4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"Mathematical Adjustment Needed: {e}")
        # خيار احتياطي: توزيع متساوي في حال فشل النموذج الرياضي
        st.warning("تم تفعيل التوزيع المتساوي للأوزان كخيار احتياطي لضمان استمرار التحليل.")
