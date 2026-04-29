import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

st.set_page_config(page_title="Quant Analysis - SA", layout="wide")

st.title("📊 منصة التحليل الكمي للأسهم السعودية")
st.sidebar.header("إعدادات المحفظة")

# اختيار الأسهم: رسن والمطاحن العربية
tickers = ['8313.SR', '2285.SR']

# تحديد التاريخ في الشريط الجانبي (مهم جداً ليعمل الكود)
start_date = st.sidebar.date_input("تاريخ بداية البيانات", value=pd.to_datetime("2024-06-15"))

# سحب البيانات
data = yf.download(tickers, start=start_date)['Close']

if not data.empty:
    st.subheader("📈 أداء الأسهم المختارة (رسن & المطاحن)")
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
        # تحويل الرموز لأسماء مفهومة عند العرض
        name = "رسن" if ticker == '8313.SR' else "المطاحن العربية"
        cols[i].metric(name, f"{weights[ticker]:.2%}")

    # إحصائيات المحفظة
    ret, vol, sharpe = ef.portfolio_performance()
    st.info(f"العائد المتوقع: {ret:.2%} | التذبذب السنوي: {vol:.2%} | نسبة شارب: {sharpe:.2f}")
else:
    st.error("لم يتم العثور على بيانات، يرجى التأكد من الرموز أو التاريخ.")
