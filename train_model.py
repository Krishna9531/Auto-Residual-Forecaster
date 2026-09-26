"""
train_model.py — Dynamic Residual Value Forecaster
===================================================
Simulates 15,000 historical vehicle transactions across 5 brands with
macroeconomic features (CPI inflation, interest rates, gas prices), trains
an XGBoost regressor, evaluates with MAPE/RMSE, and saves the model.
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# 1. Simulate realistic historical dataset
# ---------------------------------------------------------------------------
np.random.seed(42)
N = 15_000

BRANDS = {
    "Toyota":  {"models": ["Camry", "Corolla", "RAV4", "Highlander"],
                "msrp_range": (25_000, 50_000), "retention": 0.88},
    "Honda":   {"models": ["Civic", "Accord", "CR-V", "Pilot"],
                "msrp_range": (24_000, 48_000), "retention": 0.86},
    "Ford":    {"models": ["F-150", "Explorer", "Escape", "Mustang"],
                "msrp_range": (28_000, 65_000), "retention": 0.78},
    "BMW":     {"models": ["3 Series", "5 Series", "X3", "X5"],
                "msrp_range": (42_000, 85_000), "retention": 0.72},
    "Tesla":   {"models": ["Model 3", "Model Y", "Model S", "Model X"],
                "msrp_range": (40_000, 100_000), "retention": 0.80},
}

records = []
for _ in range(N):
    brand = np.random.choice(list(BRANDS.keys()))
    info = BRANDS[brand]
    model = np.random.choice(info["models"])
    msrp = np.random.uniform(*info["msrp_range"])

    # Vehicle characteristics
    age_years = np.random.uniform(0.5, 12)
    mileage = age_years * np.random.uniform(8_000, 18_000) + np.random.normal(0, 3_000)
    mileage = max(mileage, 500)

    # Macroeconomic features (realistic ranges)
    cpi_inflation = np.random.uniform(1.0, 9.0)     # %
    interest_rate = np.random.uniform(2.0, 8.0)      # %
    gas_price = np.random.uniform(2.0, 6.0)          # $/gallon

    # ---------- Residual value model (ground truth) ----------
    # Base exponential depreciation
    annual_dep_rate = 1 - info["retention"]
    base_residual = msrp * (info["retention"] ** age_years)

    # Mileage penalty: excess miles beyond 12k/yr average
    excess_miles = max(0, mileage - age_years * 12_000)
    mileage_penalty = 1 - 0.03 * (excess_miles / 10_000)
    mileage_penalty = max(mileage_penalty, 0.70)

    # Macro adjustments
    inflation_effect = 1 + 0.008 * (cpi_inflation - 3.0)       # higher CPI lifts used car prices
    interest_effect = 1 - 0.012 * (interest_rate - 4.5)         # higher rates suppress demand
    gas_effect = 1.0
    if brand == "Tesla":
        gas_effect = 1 + 0.025 * (gas_price - 3.50)            # EVs benefit from high gas prices
    elif brand in ("Ford",):
        gas_effect = 1 - 0.015 * (gas_price - 3.50)            # trucks hurt by high gas

    residual_value = (
        base_residual * mileage_penalty * inflation_effect * interest_effect * gas_effect
    )
    # Add noise (±5 %)
    residual_value *= np.random.uniform(0.95, 1.05)
    residual_value = max(residual_value, 1_000)

    records.append({
        "make": brand,
        "model": model,
        "msrp": round(msrp, 2),
        "age_years": round(age_years, 2),
        "mileage": round(mileage, 0),
        "cpi_inflation": round(cpi_inflation, 2),
        "interest_rate": round(interest_rate, 2),
        "gas_price": round(gas_price, 2),
        "residual_value": round(residual_value, 2),
    })

df = pd.DataFrame(records)
print(f"[OK] Simulated {len(df):,} vehicle transactions")
print(df.describe().round(2))

# ---------------------------------------------------------------------------
# 2. Prepare features
# ---------------------------------------------------------------------------
FEATURES = ["make", "model", "msrp", "age_years", "mileage",
            "cpi_inflation", "interest_rate", "gas_price"]
TARGET = "residual_value"

# Encode categoricals as pandas Categoricals (XGBoost native support)
df["make"] = df["make"].astype("category")
df["model"] = df["model"].astype("category")

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[DATA] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ---------------------------------------------------------------------------
# 3. Train XGBoost model
# ---------------------------------------------------------------------------
model = XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
    verbosity=1,
)

print("\n[TRAIN] Training XGBoost model ...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n[EVAL] Evaluation Results")
print(f"   RMSE : ${rmse:,.2f}")
print(f"   MAPE : {mape:.2f}%")

# ---------------------------------------------------------------------------
# 5. Save model
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "residual_model.json")
model.save_model(model_path)
print(f"\n[SAVE] Model saved -> {model_path}")
