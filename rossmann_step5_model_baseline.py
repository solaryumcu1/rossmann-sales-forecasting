import pandas as pd
import numpy as np
import xgboost as xgb

df_model = pd.read_csv("processed_train_step4_9_storeprofile_past_model_7d.csv", parse_dates=["Date"])
df_model = df_model.sort_values(["Store","Date"]).reset_index(drop=True)

VAL_DAYS = 28
max_date = df_model["Date"].max()
val_start = max_date - pd.Timedelta(days=VAL_DAYS)

train_df = df_model[df_model["Date"] < val_start].copy()
val_df   = df_model[df_model["Date"] >= val_start].copy()

TARGET = "y_7d"
DROP_COLS = ["Sales","y_7d","Date","Id"]

X_train = train_df.drop(columns=DROP_COLS, errors="ignore").select_dtypes(include=["int64","float64","bool"])
X_val   = val_df.drop(columns=DROP_COLS, errors="ignore").select_dtypes(include=["int64","float64","bool"])

y_train = train_df[TARGET].values.astype(float)
y_val   = val_df[TARGET].values.astype(float)

def rmspe(y_true, y_pred):
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask])**2))

model = xgb.XGBRegressor(
    n_estimators=8000,
    learning_rate=0.03,
    max_depth=9,
    min_child_weight=5,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=200,
    eval_metric="rmse"
)


model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

print("BEST ITER:", model.best_iteration)

pred = model.predict(X_val)
print("FINAL VALID RMSPE:", rmspe(y_val, pred))
