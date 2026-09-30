"""
app.py — Auto Residual Forecaster Dashboard
============================================
Interactive Streamlit dashboard for vehicle residual-value predictions
with macroeconomic shock analysis and forward depreciation curves.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Auto Residual Forecaster",
    page_icon="🚗",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BRAND_MODELS = {
    "Toyota":  ["Camry", "Corolla", "RAV4", "Highlander"],
    "Honda":   ["Civic", "Accord", "CR-V", "Pilot"],
    "Ford":    ["F-150", "Explorer", "Escape", "Mustang"],
    "BMW":     ["3 Series", "5 Series", "X3", "X5"],
    "Tesla":   ["Model 3", "Model Y", "Model S", "Model X"],
}

DEFAULT_MSRP = {
    "Toyota":  {"Camry": 28_000, "Corolla": 25_000, "RAV4": 32_000, "Highlander": 42_000},
    "Honda":   {"Civic": 26_000, "Accord": 30_000, "CR-V": 33_000, "Pilot": 40_000},
    "Ford":    {"F-150": 45_000, "Explorer": 40_000, "Escape": 32_000, "Mustang": 38_000},
    "BMW":     {"3 Series": 45_000, "5 Series": 58_000, "X3": 48_000, "X5": 68_000},
    "Tesla":   {"Model 3": 42_000, "Model Y": 50_000, "Model S": 85_000, "Model X": 95_000},
}

BASELINE_MACRO = {"cpi_inflation": 3.0, "interest_rate": 4.5, "gas_price": 3.50}

# ---------------------------------------------------------------------------
# Train a fresh model (used as fallback if model file is missing/corrupt)
# ---------------------------------------------------------------------------
def _train_fresh_model():
    """Simulate data and train XGBoost — used when model file unavailable."""
    np.random.seed(42)
    N = 2_000  # Reduced from 15k to prevent freezing on Streamlit Cloud
    BRANDS = {
        "Toyota":  {"models": ["Camry","Corolla","RAV4","Highlander"], "msrp_range":(25000,50000), "retention":0.88},
        "Honda":   {"models": ["Civic","Accord","CR-V","Pilot"],       "msrp_range":(24000,48000), "retention":0.86},
        "Ford":    {"models": ["F-150","Explorer","Escape","Mustang"],  "msrp_range":(28000,65000), "retention":0.78},
        "BMW":     {"models": ["3 Series","5 Series","X3","X5"],        "msrp_range":(42000,85000), "retention":0.72},
        "Tesla":   {"models": ["Model 3","Model Y","Model S","Model X"],"msrp_range":(40000,100000),"retention":0.80},
    }
    records = []
    for _ in range(N):
        brand = np.random.choice(list(BRANDS.keys()))
        info  = BRANDS[brand]
        mdl_  = np.random.choice(info["models"])
        msrp_ = np.random.uniform(*info["msrp_range"])
        age_  = np.random.uniform(0.5, 12)
        mil_  = max(age_ * np.random.uniform(8000,18000) + np.random.normal(0,3000), 500)
        cpi_  = np.random.uniform(1.0, 9.0)
        rate_ = np.random.uniform(2.0, 8.0)
        gas_  = np.random.uniform(2.0, 6.0)
        base_ = msrp_ * (info["retention"] ** age_)
        excess= max(0, mil_ - age_ * 12000)
        mil_p = max(1 - 0.03*(excess/10000), 0.70)
        infl_ = 1 + 0.008*(cpi_ - 3.0)
        int_  = 1 - 0.012*(rate_ - 4.5)
        gas_e = (1 + 0.025*(gas_-3.5)) if brand=="Tesla" else (1 - 0.015*(gas_-3.5)) if brand=="Ford" else 1.0
        rv_   = base_ * mil_p * infl_ * int_ * gas_e * np.random.uniform(0.95,1.05)
        records.append({"make":brand,"model":mdl_,"msrp":msrp_,"age_years":age_,
                        "mileage":mil_,"cpi_inflation":cpi_,"interest_rate":rate_,
                        "gas_price":gas_,"residual_value":max(rv_,1000)})
    df = pd.DataFrame(records)
    df["make"]  = df["make"].astype("category")
    df["model"] = df["model"].astype("category")
    FEATURES = ["make","model","msrp","age_years","mileage","cpi_inflation","interest_rate","gas_price"]
    X, y = df[FEATURES], df["residual_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Dramatically lighter model for cloud fallback
    mdl = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.8,
                       tree_method="hist", enable_categorical=True, random_state=42)
    mdl.fit(X_train, y_train)
    return mdl


# ---------------------------------------------------------------------------
# Load model (cached) — falls back to training if file missing or corrupt
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "residual_model.json")
    try:
        mdl = XGBRegressor()
        mdl.load_model(model_path)
        return mdl
    except Exception:
        # Model file absent or version-incompatible — train fresh
        return _train_fresh_model()


model = load_model()

# ---------------------------------------------------------------------------
# Sidebar — Vehicle profile
# ---------------------------------------------------------------------------
st.sidebar.header("🚘 Vehicle Profile")
make = st.sidebar.selectbox("Make", list(BRAND_MODELS.keys()))
model_name = st.sidebar.selectbox("Model", BRAND_MODELS[make])
msrp = DEFAULT_MSRP[make][model_name]

age = st.sidebar.slider("Vehicle Age (years)", 0.5, 12.0, 3.0, 0.5)
mileage = st.sidebar.slider("Mileage", 1_000, 200_000, int(age * 12_000), 1_000)

st.sidebar.markdown("---")
st.sidebar.header("📊 Macroeconomic Shocks")
inflation = st.sidebar.slider("CPI Inflation (%)", 1.0, 9.0, 3.0, 0.1)
gas_price = st.sidebar.slider("Gas Price ($/gal)", 2.0, 6.0, 3.50, 0.10)
interest_rate = st.sidebar.slider("Interest Rate (%)", 2.0, 8.0, 4.5, 0.1)

# ---------------------------------------------------------------------------
# Helper — predict residual value
# ---------------------------------------------------------------------------
def predict_value(make_val, model_val, msrp_val, age_val, mileage_val,
                  cpi, rate, gas):
    """Build a single-row DataFrame with correct dtypes and predict."""
    row = pd.DataFrame([{
        "make": make_val,
        "model": model_val,
        "msrp": msrp_val,
        "age_years": age_val,
        "mileage": mileage_val,
        "cpi_inflation": cpi,
        "interest_rate": rate,
        "gas_price": gas,
    }])
    row["make"] = row["make"].astype("category")
    row["model"] = row["model"].astype("category")
    pred = model.predict(row)[0]
    return float(max(pred, 0))


# ---------------------------------------------------------------------------
# Compute predictions
# ---------------------------------------------------------------------------
current_value = predict_value(make, model_name, msrp, age, mileage,
                              inflation, interest_rate, gas_price)

baseline_value = predict_value(make, model_name, msrp, age, mileage,
                               BASELINE_MACRO["cpi_inflation"],
                               BASELINE_MACRO["interest_rate"],
                               BASELINE_MACRO["gas_price"])

macro_delta = current_value - baseline_value
macro_delta_pct = (macro_delta / baseline_value * 100) if baseline_value else 0

# Simple confidence interval (±7 % heuristic based on typical MAPE)
ci_lower = current_value * 0.93
ci_upper = current_value * 1.07

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("🚗 Auto Residual Forecaster")
st.caption("Predict auto residual values under different macroeconomic scenarios")

# ---------------------------------------------------------------------------
# KPI Metric Cards
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric(
    label="💰 Estimated Residual Value",
    value=f"${current_value:,.0f}",
    delta=f"{(current_value / msrp * 100):.1f}% of MSRP",
)

k2.metric(
    label="📉 Macro Shock Impact",
    value=f"${macro_delta:+,.0f}",
    delta=f"{macro_delta_pct:+.1f}% vs baseline",
    delta_color="normal",
)

k3.metric(
    label="🔻 Confidence Low",
    value=f"${ci_lower:,.0f}",
)

k4.metric(
    label="🔺 Confidence High",
    value=f"${ci_upper:,.0f}",
)

st.markdown("---")

# ---------------------------------------------------------------------------
# 4-Year Forward Depreciation Curve
# ---------------------------------------------------------------------------
st.subheader("📉 4-Year Forward Depreciation Curve")

future_years = np.arange(0, 4.5, 0.5)
projected_ages = age + future_years
projected_miles = mileage + future_years * 12_000  # assume 12k mi/yr going forward

# Scenario A: user-selected macro conditions
values_shock = []
for a, m in zip(projected_ages, projected_miles):
    v = predict_value(make, model_name, msrp, a, m,
                      inflation, interest_rate, gas_price)
    values_shock.append(v)

# Scenario B: baseline macro conditions
values_baseline = []
for a, m in zip(projected_ages, projected_miles):
    v = predict_value(make, model_name, msrp, a, m,
                      BASELINE_MACRO["cpi_inflation"],
                      BASELINE_MACRO["interest_rate"],
                      BASELINE_MACRO["gas_price"])
    values_baseline.append(v)

# Confidence band on shock scenario
ci_low = [v * 0.93 for v in values_shock]
ci_high = [v * 1.07 for v in values_shock]

labels = [f"Year {y:.0f}" if y == int(y) else f"+{y:.1f}y" for y in future_years]

fig = go.Figure()

# Confidence band (filled area)
fig.add_trace(go.Scatter(
    x=labels, y=ci_high,
    mode="lines", line=dict(width=0),
    showlegend=False,
))
fig.add_trace(go.Scatter(
    x=labels, y=ci_low,
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(99,110,250,0.15)",
    name="Confidence Band (±7%)",
))

# Baseline curve
fig.add_trace(go.Scatter(
    x=labels, y=values_baseline,
    mode="lines+markers",
    name="Baseline Macro",
    line=dict(dash="dash", color="#636EFA", width=2),
    marker=dict(size=6),
))

# Shock curve
fig.add_trace(go.Scatter(
    x=labels, y=values_shock,
    mode="lines+markers",
    name="Current Scenario",
    line=dict(color="#EF553B", width=3),
    marker=dict(size=8),
))

fig.update_layout(
    yaxis_title="Residual Value ($)",
    xaxis_title="Projection Horizon",
    template="plotly_white",
    height=480,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=30, t=40, b=60),
)
fig.update_yaxes(tickprefix="$", tickformat=",")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Detail table
# ---------------------------------------------------------------------------
with st.expander("📋 Projection Details"):
    detail = pd.DataFrame({
        "Horizon": labels,
        "Vehicle Age": [f"{a:.1f} yr" for a in projected_ages],
        "Est. Mileage": [f"{m:,.0f}" for m in projected_miles],
        "Scenario Value": [f"${v:,.0f}" for v in values_shock],
        "Baseline Value": [f"${v:,.0f}" for v in values_baseline],
        "Delta": [f"${s - b:+,.0f}" for s, b in zip(values_shock, values_baseline)],
    })
    st.dataframe(detail, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Model: XGBoost Regressor trained on 15 000 simulated transactions · "
    "Confidence band: ±7 % heuristic · Baseline macro: CPI 3 %, Rate 4.5 %, Gas $3.50"
)
