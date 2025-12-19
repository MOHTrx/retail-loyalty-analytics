#-------------Task2-----------

import pandas as pd
import numpy as np

df = pd.read_csv("loyalty.csv")

df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0).round(2)

loyalty_order = ["0-1", "1-3", "3-5", "5-10", "10+"]
aliases = {
    "0 to 1": "0-1", "0-1 years": "0-1",
    "1 to 3": "1-3", "3 to 5": "3-5",
    "5 to 10": "5-10", "10 or more": "10+", "10+ years": "10+"
}
def norm_ly(x):
    if pd.isna(x): return "0-1"
    s = str(x).strip()
    s = aliases.get(s, s)
    return s if s in loyalty_order else "0-1"

df["loyalty_years"] = df["loyalty_years"].apply(norm_ly)

spend_by_years = (
    df.groupby("loyalty_years", as_index=False)
      .agg(avg_spend=("spend", "mean"), var_spend=("spend", "var"))
)

spend_by_years = (
    pd.DataFrame({"loyalty_years": loyalty_order})
      .merge(spend_by_years, on="loyalty_years", how="left")
)

spend_by_years["avg_spend"] = spend_by_years["avg_spend"].round(2)
spend_by_years["var_spend"] = spend_by_years["var_spend"].round(2)

spend_by_years.to_csv("spend_by_loyalty.csv", index=False)
