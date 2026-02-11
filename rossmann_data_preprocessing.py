import pandas as pd
import numpy as np

# =========================================================
# 1) LOAD (dtypewarning fix)
# =========================================================
train = pd.read_csv(
    "train.csv",
    parse_dates=["Date"],
    dtype={"StateHoliday": "string"},
    low_memory=False
)
test = pd.read_csv(
    "test.csv",
    parse_dates=["Date"],
    dtype={"StateHoliday": "string"},
    low_memory=False
)
store = pd.read_csv("store.csv")

# =========================================================
# 2) MERGE (add store-level attributes)
# =========================================================
train = train.merge(store, on="Store", how="left")
test  = test.merge(store,  on="Store", how="left")

print("After merge shapes:", train.shape, test.shape)

# =========================================================
# 3) QUICK MISSING REPORT (before)
# =========================================================
missing_before = train.isna().mean().sort_values(ascending=False)
print("\nTop missing (train, before):")
print(missing_before[missing_before > 0].head(12))

# =========================================================
# 4) BASIC CLEANING (Open + StateHoliday) with stronger logic
# =========================================================
# --- Train: Open'ı Sales'e göre doldur (garanti) ---
mask = train["Open"].isna() & (train["Sales"] > 0)
train.loc[mask, "Open"] = 1

mask = train["Open"].isna() & (train["Sales"] == 0)
train.loc[mask, "Open"] = 0

train["Open"] = train["Open"].fillna(1).astype(int)

# --- Test: Sales yok, varsayımsal 1 ---
test["Open"] = test["Open"].fillna(1).astype(int)

# StateHoliday standardize
for df in [train, test]:
    df["StateHoliday"] = df["StateHoliday"].fillna("0").astype("string").str.strip()

# Train: closed day sales -> 0 (iş kuralı tutarlılık)
train.loc[train["Open"] == 0, "Sales"] = 0

# =========================================================
# 5) MISSING HANDLING (CompetitionDistance) + extra QA flags
# =========================================================
for df in [train, test]:
    df["competition_distance_missing"] = df["CompetitionDistance"].isna().astype(int)

    med = df["CompetitionDistance"].median()
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(med)

    df["competition_date_missing_given_distance"] = (
        df["CompetitionDistance"].notna() &
        (df["CompetitionOpenSinceYear"].isna() | df["CompetitionOpenSinceMonth"].isna())
    ).astype(int)

# =========================================================
# 6) DATE FEATURES
# =========================================================
def add_date_features(df):
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dow"] = df["Date"].dt.dayofweek
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

train = add_date_features(train)
test  = add_date_features(test)

# =========================================================
# STEP-2: TURN DATES INTO DURATIONS (Competition + Promo2)
# =========================================================
def add_step2_duration_features(df):
        # ---------- Competition ----------
        # ---------- Competition (robust) ----------
    # year/month'u kopya al
    cy = df["CompetitionOpenSinceYear"]
    cm = df["CompetitionOpenSinceMonth"]

    # geçerli mask: year ve month var + year mantıklı + month mantıklı
    valid_comp = (
        cy.notna() & cm.notna() &
        (cy >= 1990) & (cy <= df["Date"].dt.year) &
        (cm >= 1) & (cm <= 12)
    )

    # has_competition: mesafe de doluysa (senin yaklaşımın) + valid tarih
    df["has_competition"] = (df["CompetitionDistance"].notna() & valid_comp).astype(int)

    # sadece valid olanlar için start date üret
    comp_start = pd.to_datetime(
        dict(
            year=cy.where(valid_comp, np.nan),
            month=cm.where(valid_comp, np.nan),
            day=1
        ),
        errors="coerce"
    )

    # ay farkı (NaT olanlarda NaN çıkar)
    months_diff = (df["Date"].dt.to_period("M") - comp_start.dt.to_period("M")).apply(lambda x: x.n if pd.notna(x) else np.nan)

    # has_competition olmayanlar 0
    df["competition_open_months"] = months_diff
    df.loc[df["has_competition"] == 0, "competition_open_months"] = 0

    # negatif/absürt durumları kırp
    df["competition_open_months"] = df["competition_open_months"].fillna(0).clip(lower=0, upper=400).astype(int)


    # ---------- Promo2 ----------
    df["Promo2"] = df["Promo2"].fillna(0).astype(int)
    df["has_promo2"] = df["Promo2"]

    p2_year = df["Promo2SinceYear"].fillna(1900).astype(int)
    p2_week = df["Promo2SinceWeek"].fillna(1).astype(int)

    promo2_start = pd.to_datetime(
    p2_year.astype(str) + "-W" + p2_week.astype(str).str.zfill(2) + "-1",
    format="%G-W%V-%u",
    errors="coerce"
)
    df["promo2_active"] = ((df["has_promo2"] == 1) & (df["Date"] >= promo2_start)).astype(int)

    weeks_active = ((df["Date"] - promo2_start).dt.days // 7)
    df["promo2_weeks_active"] = weeks_active.where(df["promo2_active"] == 1, 0).fillna(0).astype(int)

    return df

train = add_step2_duration_features(train)
test  = add_step2_duration_features(test)

print("\nStep-2 checks:")
print("has_competition rate (train):", train["has_competition"].mean())
print("competition_open_months min/max:", train["competition_open_months"].min(), train["competition_open_months"].max())
print("promo2_active rate (train):", train["promo2_active"].mean())
print("promo2_weeks_active min/max:", train["promo2_weeks_active"].min(), train["promo2_weeks_active"].max())

print("competition_open_months p99:", train["competition_open_months"].quantile(0.99))
print("promo2_weeks_active p99:", train["promo2_weeks_active"].quantile(0.99))


# =========================================================
# 7) QUICK MISSING REPORT (after Step-1 + Step-2)
# =========================================================
missing_after = train.isna().mean().sort_values(ascending=False)
print("\nTop missing (train, after Step-1+2):")
print(missing_after[missing_after > 0].head(12))

# =========================================================
# SAVE STEP-1 (opsiyonel ara çıktı)
# =========================================================
train.to_csv("processed_train_step1.csv", index=False)
test.to_csv("processed_test_step1.csv", index=False)
print("\nSaved: processed_train_step1.csv, processed_test_step1.csv")

# =========================================================
# STEP-2.5: PromoInterval -> is_promo2_month
# =========================================================
month_map = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"
}

def add_promo2_month_flags(df):
    df["PromoInterval"] = df["PromoInterval"].astype("string").fillna("").str.strip()
    df["month_str"] = df["month"].map(month_map).astype("string")

    df["is_promo2_month"] = df.apply(
        lambda r: int((r["month_str"] in r["PromoInterval"]) if r["PromoInterval"] != "" else False),
        axis=1
    )

    df["promo2_interval_count"] = df["PromoInterval"].apply(
        lambda x: 0 if x == "" else len(str(x).split(","))
    ).astype(int)

    df.drop(columns=["month_str"], inplace=True)
    return df

train = add_promo2_month_flags(train)
test  = add_promo2_month_flags(test)

print("\nStep-2.5 checks:")
print("is_promo2_month rate (train):", train["is_promo2_month"].mean())
print(train["promo2_interval_count"].value_counts().head(10))

# =========================================================
# (ARAYA) LEAKAGE DROP (Customers)
# =========================================================
for df in [train, test]:
    if "Customers" in df.columns:
        df.drop(columns=["Customers"], inplace=True)

# =========================================================
# STEP-3: CATEGORICAL ENCODING (ONE-HOT)
# =========================================================
cat_cols = ["StoreType", "Assortment", "StateHoliday"]

for df in [train, test]:
    for c in cat_cols:
        df[c] = df[c].astype("string").fillna("Unknown").str.strip()

train = pd.get_dummies(train, columns=cat_cols, drop_first=False)
test  = pd.get_dummies(test,  columns=cat_cols, drop_first=False)

train_cols = set(train.columns)
test_cols  = set(test.columns)

missing_in_test  = train_cols - test_cols
missing_in_train = test_cols - train_cols

for col in missing_in_test:
    if col != "Sales":
        test[col] = 0

for col in missing_in_train:
    train[col] = 0

feature_cols = [c for c in train.columns if c != "Sales"]
test = test[feature_cols]

print("\nAfter Step-3 (one-hot) shapes:")
print("Train:", train.shape, "Test:", test.shape)

train_features = set(train.columns) - {"Sales"}
test_features = set(test.columns)
print("Same feature count?", len(train_features) == len(test_features))

# =========================================================
# SAVE STEP-3
# =========================================================
train.to_csv("processed_train_step3.csv", index=False)
test.to_csv("processed_test_step3.csv", index=False)
print("Saved: processed_train_step3.csv, processed_test_step3.csv")

# =========================================================
# EXTRA QA PRINTS
# =========================================================
print("\nQA:")
print("Train Open value counts:\n", train["Open"].value_counts().head(5))
print("competition_distance_missing rate (train):", train["competition_distance_missing"].mean())
print("competition_date_missing_given_distance rate (train):", train["competition_date_missing_given_distance"].mean())
