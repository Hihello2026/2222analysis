import yfinance as yf
import pandas as pd

# قائمة الأسهم المختارة (أو قائمة تاسي كاملة)
tickers = ['7010.SR', '4007.SR', '2083.SR', '2285.SR']

def get_historical_yields(symbols):
    results = []
    for sym in symbols:
        ticker = yf.Ticker(sym)
        # جلب تاريخ التوزيعات
        divs = ticker.dividends
        if not divs.empty:
            # تجميع التوزيعات حسب السنة لآخر 4 سنوات
            yearly_divs = divs.groupby(divs.index.year).sum().tail(4)
            avg_div = yearly_divs.mean()
            
            # جلب السعر الحالي
            price = ticker.history(period="1d")['Close'].iloc[-1]
            
            current_yield = (avg_div / price)
            
            results.append({
                "الرمز": sym,
                "متوسط توزيع 4 سنوات": f"{avg_div:.2f} SAR",
                "سعر السهم": f"{price:.2f} SAR",
                "نسبة العائد": current_yield
            })
            
    # إنشاء الجدول وترتيبه حسب نسبة العائد تنازلياً
    df = pd.DataFrame(results)
    df = df.sort_values(by="نسبة العائد", ascending=False)
    df["نسبة العائد"] = df["نسبة العائد"].apply(lambda x: f"{x:.2%}")
    return df

print(get_historical_yields(tickers))
