import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np

# 1. إعدادات المنصة
st.set_page_config(page_title="Saudi Moat Portfolio", layout="wide")
st.title("Saudi Economic Moat Portfolio 2026")

# 2. القاموس الكامل للأسهم (الخنادق)
moat_assets = {
    '2222.SR': {'name': 'أرامكو', 'moat': 'ندرة التكلفة والاحتياطي'},
    '2223.SR': {'name': 'لوبريف', 'moat': 'تخصص زيوت الأساس'},
    '2083.SR': {'name': 'مرافق', 'moat': 'احتكار طبيعي (الجبيل وينبع)'},
    '1111.SR': {'name': 'تداول', 'moat': 'خندق تنظيمي (المشغل الوحيد)'},
    '1120.SR': {'name': 'الراجحي', 'moat': 'ودائع مجانية وهيمنة الأفراد'},
    '1180.SR': {'name': 'الأهلي', 'moat': 'تمويل المشاريع العملاقة'},
    '8313.SR': {'name': 'رسن', 'moat': 'تأثير الشبكة الرقمي'},
    '7217.SR': {'name': 'عِلم', 'moat': 'وصول حصري للبيانات الحكومية'},
    '7010.SR': {'name': 'stc', 'moat': 'بنية تحتية وبيانات ضخمة'},
    '4263.SR': {'name': 'سأل - SAL', 'moat': 'هيمنة الشحن الجوي'},
    '4031.SR': {'name': 'الخدمات الأرضية', 'moat': 'محرك المطارات التشغيلي'},
    '8210.SR': {'name': 'بوبا', 'moat': 'تخصص طبي وقوة تفاوضية'},
    '4007.SR': {'name': 'الحمادي', 'moat': 'كفاءة مالية وتمركز استراتيجي'},
    '1211.SR': {'name': 'معادن', 'moat': 'حقوق تنقيب حصرية'},
    '2020.SR': {'name': 'سابك للمغذيات', 'moat': 'كفاءة إنتاج عالمية'},
    '2200.SR': {'name': 'أنابيب السعودية', 'moat': 'مورد استراتيجي للطاقة'},
    '2280.SR': {'name': 'المراعي', 'moat': 'أضخم شبكة توزيع مبرد'},
    '1830.SR': {'name': 'وقت اللياقة', 'moat': 'سيطرة وانتشار قطاع اللياقة'}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. إعدادات المستخدم في الشريط الجانبي
portfolio_value = st.sidebar.number_input("إجمالي قيمة المحفظة (ريال)", value=1000000)
min_w = st.sidebar.slider("الحد الأدنى (%)", 1, 3, 2) / 100
max_w = st.sidebar.slider("الحد الأقصى (%)", 5, 20, 10) / 100

@st.cache_data
def get_safe_data(symbols):
    try:
        # جلب البيانات وتنظيفها فوراً من أي قيم NaN
        data = yf.download(symbols, start="2024-06-15", progress=False)['Close']
        data = data.ffill().dropna(axis=1, thresh=len(data)*0.7).dropna() # حذف الأعمدة التي تفتقر لـ 30% من البيانات
        
        actual_tickers = [t for t in symbols if t in data.columns]
        data.rename(columns=mapping, inplace=True)
        
        div_info = {}
        for sym in actual_tickers:
            ticker = yf.Ticker(sym)
            try:
                y = ticker.info.get('dividendYield', 0)
                # تصحيح الـ Scaling واستخدام 3% كبديل في حال فقدان البيانات
                div_info[mapping[sym]] = float(y) if y and 0 < y < 1 else (float(y)/100 if y and y >= 1 else 0.03)
            except:
                div_info[mapping[sym]] = 0.03
        return data, div_info
    except:
        return pd.DataFrame(), {}

price_data, dividend_yields = get_safe_data(tickers)

if not price_data.empty:
    st.subheader("تحليل الأداء السعري (الأصول المتاحة)")
    st.line_chart(price_data)

    try:
        # 4. الحسابات الرياضية مع معالجة القيم المتطرفة
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        # التأكد من عدم وجود قيم NaN في المصفوفات قبل المعالجة
        mu = mu.fillna(mu.mean())
        S = S.fillna(0)

        ef = EfficientFrontier(mu, S, weight_bounds=(min_w, max_w))
        weights = ef.max_sharpe(risk_free_rate=0.02)
        target_weights = ef.clean_weights()

        # 5. الجدول النهائي
        st.markdown("---")
        final_list = []
        total_income = 0
        
        for name, weight in target_weights.items():
            if weight > 0:
                y_rate = dividend_yields.get(name, 0.03)
                income = (weight * portfolio_value) * y_rate
                total_income += income
                
                # البحث عن الخندق من القاموس الأصلي عبر الاسم
                moat_type = next((v['moat'] for k, v in moat_assets.items() if v['name'] == name), "قوة تنافسية")
                
                final_list.append({
                    "الشركة": name,
                    "نوع الخندق": moat_type,
                    "الوزن": f"{weight:.2%}",
                    "العائد": f"{y_rate:.2%}",
                    "الدخل (ريال)": f"{income:,.2f}"
                })

        st.table(pd.DataFrame(final_list))

        # 6. المؤشرات الكلية
        st.markdown("---")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.02)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("النمو الرأسمالي المتوقع", f"{ret:.2%}")
        c2.metric("عائد التوزيعات", f"{(total_income/portfolio_value):.2%}")
        c3.metric("تذبذب المحفظة", f"{vol:.2%}")
        c4.metric("نسبة شارب", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"حدث خطأ رياضي أثناء التحسين: {e}")
        st.info("نصيحة: قد يكون ذلك بسبب قصر تاريخ تداول بعض الشركات الجديدة. حاول تقليل 'الحد الأدنى' للوزن.")
        
