import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

st.set_page_config(page_title="Quant Analysis - SA", layout="wide")

st.title("📊 منصة التحليل الكمي للأسهم السعودية")
st.sidebar.header("إعدادات المحفظة")

# اختيار الأسهم
# الرموز الجديدة: رسن (8313) والمطاحن العربية (2285)
tickers = ['8313.SR', '2285.SR']
# سحب البيانات
data = yf.download(tickers, start=start_date)['Close']

if not data.empty:
    st.subheader("📈 أداء الأسهم المختارة (مسار & المطاحن)")
    st.line_chart(data)

    # الحسابات الكمية
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S)
    weights = ef.min_volatility()
    
    # عرض النتائج
    st.subheader("⚖️ التوزيع الأمثل لتقليل المخاطر (Min Volatility)")
    cols = st.columns(len(tickers))
    for i, ticker in enumerate(tickers):
        cols[i].metric(ticker, f"{weights[ticker]:.2%}")

    # إحصائيات المحفظة
    ret, vol, sharpe = ef.portfolio_performance()
    st.info(f"العائد المتوقع: {ret:.2%} | التذبذب السنوي: {vol:.2%} | نسبة شارب: {sharpe:.2f}")
else:
    st.error("لم يتم العثور على بيانات، يرجى التأكد من الرموز أو التاريخ.")
