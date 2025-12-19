#-------------Task4-----------




import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

train_path, test_path = 'train.csv', 'test.csv'
if not (os.path.exists(train_path) and os.path.exists(test_path)):
    raise FileNotFoundError("Please upload train.csv and test.csv, then rerun.")

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

loyalty_order = ['0-1','1-3','3-5','5-10','10+']
aliases = {
    '0 to 1':'0-1','0-1 years':'0-1','1 to 3':'1-3','3 to 5':'3-5',
    '5 to 10':'5-10','10 or more':'10+','10+ years':'10+'
}
def norm_ly(x):
    if pd.isna(x): return '0-1'
    s = str(x).strip()
    s = aliases.get(s, s)
    return s if s in loyalty_order else '0-1'

def norm_month(x):
    if pd.isna(x): return 'Unknown'
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    month_map = {
        'january':'Jan','february':'Feb','march':'Mar','april':'Apr','may':'May','june':'Jun',
        'july':'Jul','august':'Aug','september':'Sep','october':'Oct','november':'Nov','december':'Dec'
    }
    s = str(x).strip().lower()
    if s in month_map: return month_map[s]
    cap = s[:3].title()
    return cap if cap in months else 'Unknown'

def norm_region(x):
    if pd.isna(x): return 'Unknown'
    valid = {'Americas','Asia/Pacific','Europe','Middle East/Africa'}
    m = {
        'asia pacific':'Asia/Pacific','asia/pacific':'Asia/Pacific',
        'middle east & africa':'Middle East/Africa','middle east/africa':'Middle East/Africa',
        'emea':'Middle East/Africa','na':'Americas','america':'Americas'
    }
    s = str(x).strip()
    s = m.get(s.lower(), s)
    return s if s in valid else 'Unknown'

def norm_promo(x):
    if pd.isna(x): return 'No'
    s = str(x).strip().lower()
    if s in {'y','yes','true','1'}: return 'Yes'
    if s in {'n','no','false','0'}: return 'No'
    return 'No'

for df in (train, test):
    for col in ['first_month']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round(2)
    df['items_in_first_month'] = (
        pd.to_numeric(df.get('items_in_first_month'), errors='coerce')
          .fillna(0).clip(lower=0).astype(int)
    )
    df['loyalty_years'] = df['loyalty_years'].apply(norm_ly)
    df['joining_month'] = df['joining_month'].apply(norm_month)
    df['region'] = df['region'].apply(norm_region)
    df['promotion'] = df['promotion'].apply(norm_promo)

train['spend'] = pd.to_numeric(train['spend'], errors='coerce').fillna(0)

feature_cols = ['first_month','items_in_first_month','region','loyalty_years','joining_month','promotion']
X_train = train[feature_cols]
y_train = train['spend']
X_test  = test[feature_cols]

numeric_features  = ['first_month','items_in_first_month']
ordinal_features  = ['loyalty_years']
nominal_features  = ['region','joining_month','promotion']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='constant', fill_value=0), numeric_features),
        ('ord', OrdinalEncoder(categories=[loyalty_order]), ordinal_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), nominal_features),
    ],
    remainder='drop'
)

model = Pipeline(steps=[
    ('prep', preprocessor),
    ('reg', LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

compare_result = pd.DataFrame({
    'customer_id': test['customer_id'],
    'spend': y_pred
})

compare_result.to_csv('compare_result.csv', index=False)
