import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# إعدادات الصفحة
st.set_page_config(page_title="Quant Analysis - SA", layout="wide")

st.title("📊 منصة التحليل الكمي للأسهم السعودية")
st.sidebar.header("إعدادات المحفظة")

# قائمة الأسهم المختارة (الراجحي، رسن، المطاحن العربية)
tickers = ['1120.SR', '8313.SR', '2285.SR']

# تحديد تاريخ البداية (يفضل منتصف 2024 لتشمل بيانات الإدراجات الجديدة)
start_date = st.sidebar.date_input("تاريخ بداية البيانات", value=pd.to_datetime("2024-06-15"))

# سحب البيانات من ياهو فاينانس
@st.cache_data
def load_data(symbols, start):
    df = yf.download(symbols, start=start)['Close']
    return df

data = load_data(tickers, start_date)

if not data.empty:
    st.subheader("📈 أداء الأسهم المختارة (الراجحي، رسن، المطاحن)")
    st.line_chart(data)

    # الحسابات الكمية باستخدام مكتبة PyPortfolioOpt
    try:
        # 1. حساب العوائد التاريخية المتوقعة
        mu = expected_returns.mean_historical_return(data)
        
        # 2. حساب مصفوفة المخاطر (التباين المشترك)
        S = risk_models.sample_cov(data)
        
        # 3. بناء الحدود الفعالة (Efficient Frontier)
        ef = EfficientFrontier(mu, S)
        
        # 4. اختيار الأوزان لتقليل التذبذب (Minimizing Volatility)
        weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()

        # عرض النتائج في واجهة الموقع
        st.subheader("⚖️ التوزيع الأمثل لتقليل المخاطر (Min Volatility)")
        cols = st.columns(len(tickers))
        
        for i, ticker in enumerate(tickers):
            if ticker == '1120.SR': name = "مصرف الراجحي"
            elif ticker == '8313.SR': name = "رسن"
            else: name = "المطاحن العربية"
            
            cols[i].metric(name, f"{cleaned_weights[ticker]:.2%}")

        # إحصائيات أداء المحفظة الإجمالية
        ret, vol, sharpe = ef.portfolio_performance()
        
        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.info(f"**العائد السنوي المتوقع:** {ret:.2%}")
        col_res2.info(f"**التذبذب (المخاطر):** {vol:.2%}")
        col_res3.success(f"**نسبة شارب (Sharpe Ratio):** {sharpe:.2f}")

    except Exception as e:
        st.error(f"حدث خطأ في الحسابات الرياضية: {e}")
else:
    st.error("لم يتم العثور على بيانات، يرجى التأكد من الرموز أو التاريخ.")

st.sidebar.markdown("""
---
**نصيحة تحليلية:**
إضافة مصرف الراجحي تعمل كـ "صمام أمان" للمحفظة بسبب استقراره العالي مقارنة بأسهم النمو الحديثة.
""")
