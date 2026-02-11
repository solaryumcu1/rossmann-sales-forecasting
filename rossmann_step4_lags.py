import pandas as pd
import numpy as np

# =========================================================
# 1) LOAD Step-3 train
# =========================================================
df = pd.read_csv("processed_train_step3.csv", parse_dates=["Date"])

# =========================================================
# 2) SORT
# =========================================================
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

# =========================================================
# 3) TARGET
# =========================================================
df["y_7d"] = df.groupby("Store")["Sales"].shift(-7)

# =========================================================
# 4) LAGS
# =========================================================
g = df.groupby("Store")["Sales"]
df["lag_7"]  = g.shift(7)
df["lag_14"] = g.shift(14)
df["lag_28"] = g.shift(28)

# =========================================================
# 5) ROLLINGS (STORE-SAFE, NO LEAKAGE)
# =========================================================
base = g.shift(1)

df["roll7_mean"]  = base.groupby(df["Store"]).rolling(7,  min_periods=7).mean().reset_index(level=0, drop=True)
df["roll14_mean"] = base.groupby(df["Store"]).rolling(14, min_periods=14).mean().reset_index(level=0, drop=True)
df["roll28_mean"] = base.groupby(df["Store"]).rolling(28, min_periods=28).mean().reset_index(level=0, drop=True)

df["roll7_std"]   = base.groupby(df["Store"]).rolling(7,  min_periods=7).std().reset_index(level=0, drop=True)
df["roll14_std"]  = base.groupby(df["Store"]).rolling(14, min_periods=14).std().reset_index(level=0, drop=True)

# =========================================================
# 6) CONTINUOUS SEASONALITY
# =========================================================
df["dayofyear"]  = df["Date"].dt.dayofyear
df["season_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
df["season_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.0)

k = 2 * np.pi / 365.0
df["season_slope"] = k * df["season_cos"]

# =========================================================
# 7) COMPETITION AGE BUCKETS
# =========================================================
if "competition_open_months" in df.columns:
    c = df["competition_open_months"]
    df["comp_age_0_1y"]  = (c <= 12).astype(int)
    df["comp_age_1_5y"]  = ((c > 12) & (c <= 60)).astype(int)
    df["comp_age_5_10y"] = ((c > 60) & (c <= 120)).astype(int)
    df["comp_age_10y+"]  = (c > 120).astype(int)
else:
    df["comp_age_0_1y"] = 0
    df["comp_age_1_5y"] = 0
    df["comp_age_5_10y"] = 0
    df["comp_age_10y+"] = 0

# =========================================================
# 8) PROMO INTERACTIONS
# =========================================================
df["promo_lag7"]  = df["Promo"] * df["lag_7"]
df["promo_roll7"] = df["Promo"] * df["roll7_mean"]

# =========================================================
# 9) MOMENTUM
# =========================================================
df["mom_7_14"]     = df["lag_7"] - df["lag_14"]
df["mom_14_28"]    = df["lag_14"] - df["lag_28"]
df["mom_roll7_14"] = df["roll7_mean"] - df["roll14_mean"]

# =========================================================
# 10) WEEKDAY ONE-HOT
# =========================================================
df["dow"] = df["Date"].dt.dayofweek
df = pd.get_dummies(df, columns=["dow"], prefix="dow", drop_first=False)

# =========================================================
# 11) STATE HOLIDAY FLAG (single numeric)
# =========================================================
state_holiday_flag = 0
for col in ["StateHoliday_a", "StateHoliday_b", "StateHoliday_c"]:
    if col in df.columns:
        state_holiday_flag = state_holiday_flag + df[col].fillna(0)
df["state_holiday_flag"] = state_holiday_flag

# =========================================================
# 12) STORE PROFILE FEATURES (PAST-ONLY, NO LEAKAGE) ✅
#     expanding mean/std computed from past sales only (shift(1))
# =========================================================
g_sales = df.groupby("Store")["Sales"]
past_sales = g_sales.shift(1)

# past-only expanding mean per store
df["store_mean_past"] = (
    past_sales.groupby(df["Store"])
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

# past-only expanding std per store (min_periods to stabilize)
df["store_std_past"] = (
    past_sales.groupby(df["Store"])
    .expanding(min_periods=30)
    .std()
    .reset_index(level=0, drop=True)
)

# fill early std NaNs safely (very early days)
df["store_std_past"] = df["store_std_past"].fillna(0)

# ratios relative to past mean (safe)
df["lag7_over_storemean_past"]  = df["lag_7"] / (df["store_mean_past"] + 1e-6)
df["roll7_over_storemean_past"] = df["roll7_mean"] / (df["store_mean_past"] + 1e-6)

# =========================================================
# PAYDAY EFFECT (DE proxy)
# =========================================================
df["dom"] = df["Date"].dt.day
df["is_payday_window"] = ((df["dom"] <= 5) | (df["dom"] >= 25)).astype(int)

# daha yumuşak: ay başı/sonu kuvveti
df["payday_strength"] = 0
df.loc[df["dom"] <= 5, "payday_strength"] = 1
df.loc[df["dom"] >= 25, "payday_strength"] = 1

# opsiyonel: ayın son günü / ilk günü
df["is_month_start"] = (df["Date"].dt.is_month_start).astype(int)
df["is_month_end"]   = (df["Date"].dt.is_month_end).astype(int)

# =========================================================
# HOLIDAY PROXIMITY (prev/next)
# =========================================================
# "holiday_any" = SchoolHoliday OR StateHoliday (a/b/c)
holiday_any = (df["SchoolHoliday"].fillna(0) == 1).astype(int)

for col in ["StateHoliday_a", "StateHoliday_b", "StateHoliday_c"]:
    if col in df.columns:
        holiday_any = np.maximum(holiday_any, df[col].fillna(0).astype(int))

df["holiday_any"] = holiday_any

# prev/next flags (store boundary safe)
df["holiday_prev1"] = df.groupby("Store")["holiday_any"].shift(1).fillna(0).astype(int)
df["holiday_next1"] = df.groupby("Store")["holiday_any"].shift(-1).fillna(0).astype(int)

# istersen daha geniş pencere:
df["holiday_prev2"] = df.groupby("Store")["holiday_any"].shift(2).fillna(0).astype(int)
df["holiday_next2"] = df.groupby("Store")["holiday_any"].shift(-2).fillna(0).astype(int)
# =========================================================
# WEEKLY SHAPE DEVIATION (past-only, leakage-safe)
# weekly_mean_past(store, dow) / store_mean_past
# =========================================================
df["dow_num"] = df["Date"].dt.dayofweek  # 0=Mon ... 6=Sun

# past-only sales for expanding stats
past_sales = df.groupby("Store")["Sales"].shift(1)

# store-level past mean
df["store_mean_past2"] = (
    past_sales.groupby(df["Store"])
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

# store+dow past mean
key = list(zip(df["Store"].values, df["dow_num"].values))
tmp = pd.Series(past_sales.values, index=pd.MultiIndex.from_tuples(key, names=["Store","dow"]))
weekly_mean_past = (
    tmp.groupby(level=[0,1])
       .expanding()
       .mean()
       .reset_index(level=[0,1], drop=True)
)

df["weekly_mean_past"] = weekly_mean_past.values

# ratio (shape)
df["weekly_shape_ratio"] = df["weekly_mean_past"] / (df["store_mean_past2"] + 1e-6)



# =========================================================
# 13) MODEL DATA (NaN cleanup)
# =========================================================
dow_cols = [c for c in df.columns if c.startswith("dow_")]

needed = [
    "y_7d",
    "lag_7", "lag_14", "lag_28",
    "roll7_mean", "roll14_mean", "roll28_mean",
    "roll7_std", "roll14_std",
    "season_sin", "season_cos", "season_slope",
    "promo_lag7", "promo_roll7",
    "mom_7_14", "mom_14_28", "mom_roll7_14",
    "state_holiday_flag",
    "store_mean_past", "store_std_past",
    "lag7_over_storemean_past", "roll7_over_storemean_past","is_payday_window", "payday_strength", "is_month_start", "is_month_end",
    "holiday_any", "holiday_prev1", "holiday_next1", "holiday_prev2", "holiday_next2",
    "weekly_shape_ratio",
] + dow_cols

df_model = df.dropna(subset=needed).copy()
if "Open" in df_model.columns:
    df_model = df_model[df_model["Open"] == 1].copy()

print("Before drop:", df.shape)
print("After  drop:", df_model.shape)
print("Num features added (dow):", len(dow_cols))

# =========================================================
# 14) SAVE
# =========================================================
out_path = "processed_train_step4_9_storeprofile_past_model_7d.csv"
df_model.to_csv(out_path, index=False)
print("Saved:", out_path)
