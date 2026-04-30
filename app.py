import streamlit as st
import yfinance as yf
import pandas as pd

# إعداد واجهة المنصة
st.set_page_config(page_title="Strategic Portfolio Growth & Yield", layout="wide")
st.title("Strategic Equity Analysis: Growth vs. Yield")

# تعريف الكون الاستثماري المحدث
tickers = ['7010.SR', '4007.SR', '2285.SR', '7203.SR', '2083.SR', '1111.SR']
mapping = {
    '7010.SR': 'stc',
    '4007.SR': 'Al Hammadi',
    '2285.SR': 'Arabian Mills',
    '7203.SR': 'ELM',
    '2083.SR': 'Marafiq',
    '1111.SR': 'Tadawul'
}

# إدخالات المستخدم
portfolio_value = st.sidebar.number_input("Total Portfolio Value (SAR)", value=1000000)

@st.cache_data
def get_professional_data(symbols):
    df = yf.download(symbols, start="2024-06-15")['Close']
    df.rename(columns=mapping, inplace=True)
    divs = {}
    for sym in symbols:
        ticker = yf.Ticker(sym)
        y = ticker.info.get('dividendYield', 0)
        divs[mapping[sym]] = float(y) if y and y < 1 else (float(y)/100 if y else 0)
    return df, divs

price_data, dividend_yields = get_professional_data(tickers)

# التوزيع المقترح يدوياً لتحقيق التوازن (Manual Strategic Allocation)
# بدلاً من ترك الخوارزمية تختار 80% لسهم واحد، نضع أوزان استراتيجية
balanced_weights = {
    'stc': 0.30,           # قاعدة العوائد
    'ELM': 0.20,           # نمو تقني
    'Tadawul': 0.15,       # نمو مالي
    'Marafiq': 0.15,       # عوائد مرافق
    'Al Hammadi': 0.10,    # نمو ودخل صحي
    'Arabian Mills': 0.10  # قيمة استهلاكية
}

if not price_data.empty:
    st.subheader("Automated Management & Yield Analysis")
    mgmt_data = []
    total_income = 0
    
    for asset, weight in balanced_weights.items():
        y_rate = dividend_yields.get(asset, 0)
        income = (weight * portfolio_value) * y_rate
        total_income += income
        
        mgmt_data.append({
            "Asset": asset,
            "Strategy": "Growth" if asset in ['ELM', 'Tadawul'] else "Income/Value",
            "Weight %": f"{weight:.2%}",
            "Div. Yield": f"{y_rate:.2%}",
            "Est. Income (SAR)": f"{income:,.2f}"
        })
    
    st.table(pd.DataFrame(mgmt_data))

    # ملخص الأداء العام
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Portfolio Yield", f"{(total_income/portfolio_value):.2%}")
    m2.metric("Total Est. Income", f"SAR {total_income:,.2f}")
    m3.metric("Growth Exposure", "35.00%")
