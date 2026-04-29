import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# إعدادات الصفحة
st.set_page_config(page_title="Quant Analysis - High Growth", layout="wide")

st.title("🚀 منصة التحليل الكمي - استراتيجية العوائد المرتفعة")
st.sidebar.header("إعدادات المحفظة الهجومية")

# قائمة الأسهم المختارة (stc، رسن، المطاحن العربية، علم)
# سهم 'علم' (7203) هو المحرك الرئيسي لرفع العائد
tickers = ['7010.SR', '8313.SR', '2285.SR', '7203.SR']

# تاريخ البداية
start_date = st.sidebar.date_input("تاريخ بداية البيانات", value=pd.to_datetime("2024-06-15"))

# سحب البيانات
@st.cache_data
def load_data(symbols, start):
    df = yf.download(symbols, start=start)['Close']
    return df

data = load_data(tickers, start_date)

if not data.empty:
    st.subheader("📈 أداء الأسهم الهجومية (stc، رسن، المطاحن، علم)")
    st.line_chart(data)

    try:
        # الحسابات الكمية المتقدمة
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        # بناء الحدود الفعالة
        ef = EfficientFrontier(mu, S)
        
        # استراتيجية تعظيم نسبة شارب للوصول لعائد من خانتين
        weights = ef.max_sharpe() 
        cleaned_weights = ef.clean_weights()

        # عرض التوزيع المقترح
        st.subheader("⚖️ التوزيع الأمثل لتحقيق أعلى كفاءة (Max Sharpe)")
        cols = st.columns(len(tickers))
        
        for i, ticker in enumerate(tickers):
            if ticker == '7010.SR': name = "stc"
            elif ticker == '8313.SR': name = "رسن"
            elif ticker == '2285.SR': name = "المطاحن العربية"
            else: name = "علم (ELM)"
            
            cols[i].metric(name, f"{cleaned_weights[ticker]:.2%}")

        # إحصائيات الأداء المستهدف
        ret, vol, sharpe = ef.portfolio_performance()
        
        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)
        
        # عرض العائد بلون بارز إذا تجاوز 10%
        ret_label = "العائد السنوي المتوقع"
        if ret >= 0.10:
            col_res1.success(f"**{ret_label}: {ret:.2%}** ✅")
        else:
            col_res1.info(f"**{ret_label}: {ret:.2%}**")
            
        col_res2.warning(f"**التذبذب (المخاطر): {vol:.2%}**")
        col_res3.success(f"**نسبة شارب: {sharpe:.2f}**")

    except Exception as e:
        st.error(f"حدث خطأ في الحسابات: {e}")
        st.info("نصيحة: إذا ظهر خطأ، جرب تغيير تاريخ البداية ليغطي فترة أطول.")
else:
    st.error("لم يتم العثور على بيانات.")

st.sidebar.markdown("""
---
**تحليل الاستراتيجية:**
هذه النسخة لا تكتفي بتقليل المخاطر، بل تبحث عن "النقطة السحرية" التي تعطي أعلى عائد ممكن. سهم **علم** و **stc** يمثلان توازناً ممتازاً بين النمو السريع والأمان المالي.
""")
