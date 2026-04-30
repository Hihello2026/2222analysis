try:
    # ... (الأجزاء السابقة من الكود)
    
    # بناء الحدود الفعالة
    ef = EfficientFrontier(mu, S)
    
    # إضافة القيد الجديد: فرض حد أدنى 5% لكل سهم لضمان توزيع الـ 20% الضائعة
    # هذا يضمن أن "المطاحن"، "تداول"، و"المرافق" لن تكون أوزانها صفراً
    ef.add_constraint(lambda w: w >= 0.05) 
    
    # تحسين نسبة شارب مع القيود الجديدة
    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()

    # ... (باقي الكود للعرض)
