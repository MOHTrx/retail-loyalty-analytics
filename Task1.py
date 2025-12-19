#-------------Task1-----------

import pandas as pd
import numpy as np

df = pd.read_csv("loyalty.csv")

expected_cols = [
    "customer_id", "spend", "first_month", "items_in_first_month",
    "region", "loyalty_years", "joining_month", "promotion"
]
df = df.rename(columns={
    "items_first_month": "items_in_first_month",
    "items_in_1st_month": "items_in_first_month",
    "first_month_spend": "first_month"
})
for col in expected_cols:
    if col not in df.columns:
        df[col] = np.nan
df = df[expected_cols]

for col in ["spend", "first_month"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(2)

df["items_in_first_month"] = (
    pd.to_numeric(df["items_in_first_month"], errors="coerce")
      .fillna(0).clip(lower=0).astype(int)
)

valid_regions = {"Americas", "Asia/Pacific", "Europe", "Middle East/Africa"}
region_map = {
    "asia pacific": "Asia/Pacific",
    "asia/pacific": "Asia/Pacific",
    "middle east & africa": "Middle East/Africa",
    "middle east/africa": "Middle East/Africa",
    "emea": "Middle East/Africa",
    "na": "Americas",
    "america": "Americas",
}
def normalize_region(x):
    if pd.isna(x): return "Unknown"
    s = str(x).strip()
    s = region_map.get(s.lower(), s)
    return s if s in valid_regions else "Unknown"
df["region"] = df["region"].apply(normalize_region)

loyalty_order = ["0-1", "1-3", "3-5", "5-10", "10+"]
loyalty_alias = {
    "0 to 1": "0-1", "0-1 years": "0-1",
    "1 to 3": "1-3", "3 to 5": "3-5",
    "5 to 10": "5-10", "10 or more": "10+", "10+ years": "10+"
}
def normalize_loyalty(x):
    if pd.isna(x): return "0-1"
    s = str(x).strip()
    s = loyalty_alias.get(s, s)
    return s if s in loyalty_order else "0-1"
df["loyalty_years"] = pd.Categorical(
    df["loyalty_years"].apply(normalize_loyalty),
    categories=loyalty_order, ordered=True
)

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_map = {
    "january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr", "may": "May",
    "june": "Jun", "july": "Jul", "august": "Aug", "september": "Sep",
    "october": "Oct", "november": "Nov", "december": "Dec"
}
def normalize_month(x):
    if pd.isna(x): return "Unknown"
    s = str(x).strip()
    if s.lower() in month_map:
        return month_map[s.lower()]
    cap = s[:3].title()
    return cap if cap in months else "Unknown"
df["joining_month"] = df["joining_month"].apply(normalize_month)

def normalize_promo(x):
    if pd.isna(x): return "No"
    s = str(x).strip().lower()
    if s in {"y","yes","true","1"}: return "Yes"
    if s in {"n","no","false","0"}: return "No"
    return "No"
df["promotion"] = df["promotion"].apply(normalize_promo)

clean_data = df.copy()

clean_data.to_csv("loyalty_clean.csv", index=False)
