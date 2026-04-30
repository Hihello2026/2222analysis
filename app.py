import streamlit as st
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

# 1. إعدادات الصفحة الاحترافية
st.set_page_config(page_title="Quantitative Equity Analysis", layout="wide")

# تصميم واجهة مؤسسية (Industrial Minimalism)
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

# 2. تعريف الأصول والبيانات الاحتياطية (تجنباً لظهور نسبة 0% في العوائد)
tickers = ['7010.SR', '4007.SR', '2285.SR', '2083.SR']
mapping = {
    '7010.SR': 'stc',
    '4007.SR': 'Al Hammadi',
    '2285.SR': 'Arabian Mills',
    '2083.SR': 'Marafiq'
}

# قيم عوائد التوزيعات التقريبية لعام 2026 كمصدر احتياطي
fallback_yields = {
    '7010.SR': 0.0516,  # stc
    '4007.SR': 0.0422,  # الحمادي
    '2083.SR': 0.0525,  # المرافق
    '2285.SR': 0.0243   # المطاحن العربية
}

# 3. مدخلات المستخدم
start_date = st.sidebar.date_input("Analysis Start Date", value=pd.to_datetime("2024-06-15"))
portfolio_value = st.sidebar.number_input("Total Portfolio Value (SAR)", min_value=1000, value=1000000)
risk_free_rate = 0.02

@st.cache_data(show_spinner="جاري جلب بيانات السوق وتحليل العوائد...")
def get_portfolio_data(symbols, start):
    try:
        # جلب أسعار الإغلاق
        price_df = yf.download(symbols, start=start, progress=False)['Close']
        if price_df.empty:
            return pd.DataFrame(), {}
            
        price_df.rename(columns=mapping, inplace=True)
        
        div_info = {}
        for sym in symbols:
            ticker_obj = yf.Ticker(sym)
            try:
                # محاولة جلب العائد الحي
                y_val = ticker_obj.info.get('dividendYield')
                if y_val and y_val > 0:
                    div_info[mapping[sym]] = float(y_val) if y_val < 1 else float(y_val) / 100
                else:
                    # استخدام القيمة الاحتياطية إذا كان العائد 0 أو مفقوداً
                    div_info[mapping[sym]] = fallback_yields.get(sym, 0.0)
            except:
                div_info[mapping[sym]] = fallback_yields.get(sym, 0.0)
                
        return price_df, div_info
    except Exception as e:
        st.error(f"خطأ في الاتصال بالبيانات: {e}")
        return pd.DataFrame(), {}

# جلب البيانات
price_data, dividend_yields = get_portfolio_data(tickers, start_date)

if not price_data.empty:
    # 4. عرض أداء الأسعار (Chart)
    st.subheader("Asset Price Performance (SAR)")
    st.line_chart(price_data)

    try:
        # 5. الحسابات الكمية (Mean-Variance Optimization)
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        # تحسين المحفظة بناءً على أقصى نسبة شارب
        ef = EfficientFrontier(mu, S)
        # فرض حد أدنى للتنويع (5%) لكل سهم وحد أقصى (60%) لتجنب التركيز العالي جداً
        ef.add_constraint(lambda w: w >= 0.05)
        ef.add_constraint(lambda w: w <= 0.60) 
        
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate) 
        target_weights = ef.clean_weights()

        # 6. جدول إدارة المحفظة وتحليل العوائد
        st.markdown("---")
        st.subheader("Portfolio Management & Yield Analysis")
        
        mgmt_data = []
        total_income = 0
        
        for asset, t_weight in target_weights.items():
            y_rate = dividend_yields.get(asset, 0)
            asset_sar_value = t_weight * portfolio_value
            annual_income = asset_sar_value * y_rate
            total_income += annual_income
            
            mgmt_data.append({
                "Asset": asset,
                "Target Weight": f"{t_weight:.2%}",
                "Div. Yield": f"{y_rate:.2%}",
                "Est. Annual Income (SAR)": f"{annual_income:,.2f}",
                "Action": "Optimal"
            })

        st.table(pd.DataFrame(mgmt_data))

        # 7. ملخص الأداء المؤسسي
        st.markdown("---")
        st.subheader("Institutional Performance Metrics")
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Cap. Gain", f"{ret:.2%}")
        m2.metric("Portfolio Yield", f"{(total_income/portfolio_value):.2%}")
        m3.metric("Annual Volatility", f"{vol:.2%}")
        m4.metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.sidebar.success("تم تحديث المحفظة بنجاح مع تفعيل آلية التصحيح التلقائي للعوائد.")

    except Exception as e:
        st.error(f"خطأ في العمليات الحسابية: {e}")
else:
    st.warning("⚠️ لا توجد بيانات كافية للتحليل. يرجى مراجعة رموز الأسهم أو الاتصال بالشبكة.")
