import streamlit as st  # هذا السطر يجب أن يكون أول سطر في الملف
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np
from datetime import datetime
import requests

# الآن يمكنك استخدام st.cache_data بأمان
@st.cache_data(ttl=3600)
def load_market_data(symbols, start):
    try:
        # الكود الخاص بجلب البيانات كما أعددناه سابقاً
        raw = yf.download(symbols + ['^TASI.SR'], start=start, progress=False)['Close']
        
        # آلية الدفاع ضد الأخطاء أو أيام الإجازات
        if raw.empty or len(raw) < 5:
            adjusted_start = (pd.to_datetime(start) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
            raw = yf.download(symbols + ['^TASI.SR'], start=adjusted_start, progress=False)['Close']

        if raw.empty: return pd.DataFrame(), pd.Series()
        
        data = raw.ffill().dropna()
        benchmark = data['^TASI.SR']
        assets = data.drop(columns=['^TASI.SR'])
        return assets, benchmark
    except Exception as e:
        return pd.DataFrame(), pd.Series()

# استكمال باقي كود التطبيق (الهوية، الأسهم، التنبيهات...)
