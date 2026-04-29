import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# إعدادات الصفحة
st.set_page_config(page_title="Quant Analysis - SA", layout="wide")

st.title("📊 منصة التحليل الكمي للأسهم السعودية")
st.sidebar.header("إعدادات المحفظة")

# قائمة الأسهم الجديدة (stc، رسن، المطاحن العربية، مرافق)
tickers = ['7010.SR', '8313.SR', '2285.SR', '2083.SR']

# تحديد تاريخ البداية
start_date = st.sidebar.date_input("تاريخ بداية البيانات", value=pd.to_datetime("2024-06-15"))

# سحب البيانات
@st.cache_data
def load_data(symbols, start):
    df = yf.download(symbols, start=start)['Close']
    return df

data = load_data(tickers, start_date)

if not data.empty:
    st.subheader("📈 أداء الأسهم المختارة (stc، رسن، المطاحن، مرافق)")
    st.line_chart(data)

    try:
        # الحسابات الكمية
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        
        # اختيار الأوزان لتقليل التذبذب
        weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()

        # عرض النتائج في واجهة الموقع
        st.subheader("⚖️ التوزيع الأمثل لتقليل المخاطر (Min Volatility)")
        cols = st.columns(len(tickers))
        
        for i, ticker in enumerate(tickers):
            if ticker == '7010.SR': name = "stc"
            elif ticker == '8313.SR': name = "رسن"
            elif ticker == '2285.SR': name = "المطاحن العربية"
            else: name = "مرافق"
            
            cols[i].metric(name, f"{cleaned_weights[ticker]:.2%}")

        # إحصائيات أداء المحفظة
        ret, vol, sharpe = ef.portfolio_performance()
        
        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.info(f"**العائد السنوي المتوقع:** {ret:.2%}")
        col_res2.info(f"**التذبذب (المخاطر):** {vol:.2%}")
        
        # تلوين النتيجة بناءً على نسبة شارب
        if sharpe >= 0:
            col_res3.success(f"**نسبة شارب (Sharpe Ratio):** {sharpe:.2f}")
        else:
            col_res3.error(f"**نسبة شارب (Sharpe Ratio):** {sharpe:.2f}")

    except Exception as e:
        st.error(f"حدث خطأ في الحسابات الرياضية: {e}")
else:
    st.error("لم يتم العثور على بيانات، يرجى التأكد من الرموز أو التاريخ.")

st.sidebar.markdown("""
---
**تحليل المحفظة:**
تعتبر **stc** الركيزة الأساسية للاستقرار، بينما توفر **مرافق** حماية قطاعية، وتضيف **رسن** و**المطاحن** فرص نمو متوازنة.
""")
