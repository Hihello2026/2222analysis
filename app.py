@st.cache_data(ttl=3600) # تحديث البيانات كل ساعة بدلاً من الاعتماد التام على الكاش
def load_data(symbols):
    try:
        # إضافة ^TASI.SR لضمان وجود المؤشر العام للمقارنة
        full_tickers = symbols + ['^TASI.SR']
        
        # محاولة سحب البيانات مع تحديد خيار auto_adjust لضمان الدقة
        raw_data = yf.download(
            full_tickers, 
            start="2024-01-01", 
            end=datetime.now().strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=True
        )['Close']
        
        if raw_data.empty:
            st.error("السيرفر لم يرسل أي بيانات. جرب تغيير تاريخ البدء.")
            return pd.DataFrame(), pd.Series()
            
        # تنظيف البيانات
        data = raw_data.ffill().dropna()
        
        # التأكد من وجود المؤشر والأسهم بعد التنظيف
        if '^TASI.SR' in data.columns:
            benchmark = data['^TASI.SR']
            assets = data.drop(columns=['^TASI.SR']).rename(columns=mapping)
            return assets, benchmark
        else:
            return data.rename(columns=mapping), pd.Series()
            
    except Exception as e:
        st.warning(f"Connection Issue: {e}")
        return pd.DataFrame(), pd.Series()
