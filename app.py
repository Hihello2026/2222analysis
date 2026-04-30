import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# 1. إعدادات المنصة
st.set_page_config(page_title="Saudi Moat Portfolio", layout="wide")
st.title("Saudi Economic Moat Portfolio 2026")
st.markdown("تحليل المحفظة القائم على الشركات ذات المزايا التنافسية المستدامة (Economic Moats)")

# 2. تعريف الأصول والخنادق (القاموس الشامل)
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

# 3. إعدادات المحفظة
portfolio_value = st.sidebar.number_input("إجمالي قيمة المحفظة (ريال)", value=1000000)
min_weight = st.sidebar.slider("الحد الأدنى لكل سهم (%)", 1, 5, 2) / 100
max_weight = st.sidebar.slider("الحد الأقصى لكل سهم (%)", 10, 30, 15) / 100

@st.cache_data
def get_moat_data(symbols):
    try:
        df = yf.download(symbols, start="2024-01-01", progress=False)['Close']
        df.rename(columns=mapping, inplace=True)
        
        div_info = {}
        for sym in symbols:
            ticker = yf.Ticker(sym)
            y = ticker.info.get('dividendYield', 0)
            # تصحيح Scaling Error
            div_info[mapping[sym]] = float(y) if y and y < 1 else (float(y)/100 if y else 0.03) # 3% كافتراضي
        return df, div_info
    except:
        return pd.DataFrame(), {}

price_data, dividend_yields = get_moat_data(tickers)

if not price_data.empty:
    st.subheader("الأداء السعري لشركات الخنادق الاستراتيجية")
    st.line_chart(price_data)

    try:
        # 4. التحسين الكمي (Mean-Variance Optimization)
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
        weights = ef.max_sharpe(risk_free_rate=0.02)
        target_weights = ef.clean_weights()

        # 5. جدول إدارة المحفظة الشامل
        st.markdown("---")
        st.subheader("توزيع الأصول بناءً على القوة التنافسية والعوائد")
        
        final_table = []
        total_income = 0
        
        for ticker, info in moat_assets.items():
            name = info['name']
            weight = target_weights.get(name, 0)
            y_rate = dividend_yields.get(name, 0)
            income = (weight * portfolio_value) * y_rate
            total_income += income
            
            final_table.append({
                "الشركة": name,
                "نوع الخندق الاقتصادي": info['moat'],
                "الوزن المقترح": f"{weight:.2%}",
                "عائد التوزيع": f"{y_rate:.2%}",
                "الدخل السنوي المتوقع": f"{income:,.2f}"
            })

        st.table(pd.DataFrame(final_table))

        # 6. مؤشرات الأداء الكلية
        st.markdown("---")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.02)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("نمو رأس المال المتوقع", f"{ret:.2%}")
        col2.metric("عائد المحفظة النقدي", f"{(total_income/portfolio_value):.2%}")
        col3.metric("مستوى التذبذب (المخاطرة)", f"{vol:.2%}")
        col4.metric("نسبة شارب (كفاءة العائد)", f"{sharpe:.2f}")

    except Exception as e:
        st.error(f"خطأ في الحسابات: {e}")
