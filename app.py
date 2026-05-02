@st.cache_data(ttl=3600)
def load_market_data(symbols, start):
    try:
        # المحاولة الأولى: جلب البيانات العادية
        raw = yf.download(symbols + ['^TASI.SR'], start=start, progress=False)['Close']
        
        # إذا كانت البيانات فارغة (بسبب يوم الإجازة أو عطل السيرفر)
        if raw.empty or len(raw) < 5:
            # المحاولة الثانية: توسيع نطاق البحث قليلاً للخلف لضمان وجود بيانات
            adjusted_start = (pd.to_datetime(start) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
            raw = yf.download(symbols + ['^TASI.SR'], start=adjusted_start, progress=False)['Close']

        if raw.empty: return pd.DataFrame(), pd.Series()
        
        data = raw.ffill().dropna()
        benchmark = data['^TASI.SR']
        assets = data.drop(columns=['^TASI.SR']).rename(columns=mapping)
        return assets, benchmark
    except:
        return pd.DataFrame(), pd.Series()
        
