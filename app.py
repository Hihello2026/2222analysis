import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np

# 1. إعدادات المنصة
st.set_page_config(page_title="Saudi Moat Portfolio", layout="wide")
st.title("Saudi Economic Moat Portfolio 2026")
st.markdown("تم تحديث المحفظة: إضافة قطاع الطاقة الاستراتيجي بدلاً من التأمين الطبي")

# 2. القاموس المحدث للأسهم (تم استبدال بوبا بـ السعودية للطاقة)
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
    '2381.SR': {'name': 'الحفر العربية', 'moat': 'العمود الفقري لخدمات الطاقة (السعودية للطاقة)'}, # تم الاستبدال هنا
    '4007.SR': {'name': 'الحمادي', 'moat': 'كفاءة مالية وتمركز استراتيجي'},
    '1211.SR': {'name': 'معادن', 'moat': 'حقوق تنقيب حصرية'},
    '2020.SR': {'name': 'سابك للمغذيات', 'moat': 'كفاءة إنتاج عالمية'},
    '2200.SR': {'name': 'أنابيب السعودية', 'moat': 'مورد استراتيجي للطاقة'},
    '2280.SR': {'name': 'المراعي', 'moat': 'أضخم شبكة توزيع مبرد'},
    '1830.SR': {'name': 'وقت اللياقة', 'moat': 'سيطرة وانتشار قطاع اللياقة'}
}

tickers = list(moat_assets.keys())
mapping = {k: v['name'] for k, v in moat_assets.items()}

# 3. إعدادات المستخدم
portfolio_value = st.sidebar.number_input("إجمالي قيمة المحفظة (ريال)", value=1000000)
min_w = st.sidebar.slider("الحد الأدنى (%)", 1, 3, 2) / 100
max_w = st.sidebar.slider("الحد الأقصى (%)", 5, 25, 12) / 100

@st.cache_data
def get_safe_data(symbols):
    try:
        # جلب البيانات وتنظيفها
        data = yf.download(symbols, start="2024-06-15", progress=False)['Close']
        data = data.ffill().dropna(axis=1, thresh=len(data)*0.5).dropna()
        
        actual_tickers = [t for t in symbols if t in data.columns]
        data.rename(columns=mapping, inplace=True)
        
        div_info = {}
        # قيم افتراضية للعوائد بناءً على التوقعات المالية 2026
        fallback = {'stc': 0.0516, 'أرامكو': 0.045, 'الحمادي': 0.0307, 'الحفر العربية': 0.035}
        
        for sym in actual_tickers:
            name = mapping[sym]
            ticker = yf.Ticker(sym)
            try:
                y = ticker.info.get('dividendYield', 0)
                div_info[name] = float(y) if y and 0 < y < 1 else (float(y)/100 if y and y >= 1 else fallback.get(name, 0.03))
            except:
                div_info[name] = fallback.get(name, 0.03)
        return data, div_info
    except:
        return pd.DataFrame(), {}

price_data, dividend_yields = get_safe_data(tickers)

if not price_data.empty:
    st.subheader("الأداء السعري الموحد لشركات الخنادق الاستراتيجية")
    st.line_chart(price_data)

    try:
        # 4. التحسين الرياضي
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        # ضبط المصفوفات لتجنب NaN
        mu = mu.fillna(mu.mean())
        S = S.fillna(0)

        ef = EfficientFrontier(mu, S, weight_bounds=(min_w, max_w))
        weights = ef.max_sharpe(risk_free_rate=0.02)
        target_weights = ef.clean_weights()

        # 5. عرض النتائج
        st.markdown("---")
        final_list = []
        total_income = 0
        
        for name, weight in target_weights.items():
            if weight > 0.001:
                y_rate = dividend_yields.get(name, 0.03)
                income = (weight * portfolio_value) * y_rate
                total_income += income
                
                moat_type = next((v['moat'] for k, v in moat_assets.items() if v['name'] == name), "قوة تنافسية")
                
                final_list.append({
                    "الشركة": name,
                    "الخندق الاقتصادي": moat_type,
                    "الوزن": f"{weight:.2%}",
                    "العائد النقدى": f"{y_rate:.2%}",
                    "الدخل المتوقع": f"{income:,.2f}"
                })

        st.table(pd.DataFrame(final_list))

        # 6. المؤشرات الكلية
        st.markdown("---")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.02)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("نمو رأس المال المتوقع", f"{ret:.2%}")
        c2.metric("عائد التوزيعات النقدي", f"{(total_income/portfolio_value):.2%}")
        c3.metric("تذبذب المحفظة", f"{vol:.2%}")
        c4.metric("نسبة شارب", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"حدث خطأ في معالجة البيانات: {e}")
