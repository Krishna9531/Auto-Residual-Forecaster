"""
app.py — Auto Residual Forecaster Dashboard (Brutalist Edition)
================================================================
Advanced interactive Streamlit dashboard for vehicle residual-value predictions
with macroeconomic shock analysis, loan equity tracking, and a Brutalist UI.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Page configuration & BRUTALIST CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AUTO_RESIDUAL_FORECASTER_V2",
    page_icon="⚠️",
    layout="wide",
)

st.markdown("""
<style>
    /* BRUTALIST UI CORE */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Mono', monospace !important;
        background-color: #E8E8E8 !important;
        color: #000000 !important;
    }
    
    /* Harsh borders and shadows for structural elements */
    .stApp {
        background-color: #E8E8E8;
        background-image: radial-gradient(#000000 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    .stSidebar, [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 6px solid #000000 !important;
    }
    
    h1, h2, h3 {
        text-transform: uppercase;
        font-weight: 900 !important;
        letter-spacing: -2px;
        background: #000000;
        color: #E6FF00 !important;
        padding: 10px 15px;
        border: 4px solid #000000;
        box-shadow: 6px 6px 0px #FF00FF;
        display: inline-block;
        margin-bottom: 20px !important;
    }
    
    /* Metrics blocks */
    div[data-testid="metric-container"] {
        background: #FFFFFF;
        border: 4px solid #000000;
        padding: 15px;
        box-shadow: 8px 8px 0px #000000;
        border-radius: 0px !important;
        transition: transform 0.1s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translate(-2px, -2px);
        box-shadow: 10px 10px 0px #FF00FF;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 3.5rem !important;
        font-weight: 900;
        text-shadow: 3px 3px 0px #00FFFF;
    }
    
    /* Inputs and Buttons */
    .stSelectbox div[data-baseweb="select"] > div, 
    .stSlider > div > div > div > div {
        border: 3px solid #000000 !important;
        border-radius: 0px !important;
        background-color: #FFFFFF !important;
        box-shadow: 4px 4px 0px #000000 !important;
    }
    
    button {
        background-color: #E6FF00 !important;
        color: #000000 !important;
        border: 4px solid #000000 !important;
        box-shadow: 6px 6px 0px #000000 !important;
        border-radius: 0px !important;
        text-transform: uppercase;
        font-weight: 900 !important;
        font-size: 1.2rem !important;
        padding: 10px 20px !important;
    }
    button:active {
        transform: translate(6px, 6px) !important;
        box-shadow: 0px 0px 0px #000000 !important;
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        background-color: #FFFFFF;
        border: 3px solid #000;
        margin-right: 10px;
        border-radius: 0 !important;
        box-shadow: 4px 4px 0px #000;
        font-weight: 900 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #FF00FF !important;
        color: #FFFFFF !important;
    }
    
    /* General elements */
    hr {
        border: none;
        border-top: 6px dashed #000000;
        margin: 40px 0;
    }
    .stDataFrame {
        border: 4px solid #000000;
        box-shadow: 6px 6px 0px #00FFFF;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants & Helpers
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

def get_brutalist_plotly_layout(title=""):
    return dict(
        title=dict(text=f"<b>{title}</b>", font=dict(family="Space Mono", size=20, color="#000")),
        font_family="Space Mono, monospace",
        font_color="#000",
        plot_bgcolor="#FFF",
        paper_bgcolor="#E8E8E8",
        margin=dict(l=50, r=30, t=60, b=50),
        xaxis=dict(
            showgrid=True, gridcolor="#000", gridwidth=2, griddash="dot",
            zeroline=True, zerolinewidth=4, zerolinecolor="#000",
            showline=True, linewidth=4, linecolor="#000", mirror=True
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#000", gridwidth=2, griddash="dot",
            zeroline=True, zerolinewidth=4, zerolinecolor="#000",
            showline=True, linewidth=4, linecolor="#000", mirror=True
        ),
        hovermode="x unified"
    )

# ---------------------------------------------------------------------------
# Fallback Model Generator (Ultra-Fast)
# ---------------------------------------------------------------------------
def _train_fresh_model():
    """Simulate data and train XGBoost — used when model file unavailable."""
    np.random.seed(42)
    N = 2_000
    records = []
    for _ in range(N):
        brand = np.random.choice(list(BRAND_MODELS.keys()))
        info_retention = {"Toyota":0.88, "Honda":0.86, "Ford":0.78, "BMW":0.72, "Tesla":0.80}[brand]
        mdl_  = np.random.choice(BRAND_MODELS[brand])
        msrp_ = np.random.uniform(24000, 100000)
        age_  = np.random.uniform(0.5, 12)
        mil_  = max(age_ * np.random.uniform(8000,18000) + np.random.normal(0,3000), 500)
        cpi_  = np.random.uniform(1.0, 9.0)
        rate_ = np.random.uniform(2.0, 8.0)
        gas_  = np.random.uniform(2.0, 6.0)
        
        base_ = msrp_ * (info_retention ** age_)
        excess= max(0, mil_ - age_ * 12000)
        mil_p = max(1 - 0.03*(excess/10000), 0.70)
        infl_ = 1 + 0.008*(cpi_ - 3.0)
        int_  = 1 - 0.012*(rate_ - 4.5)
        gas_e = (1 + 0.025*(gas_-3.5)) if brand=="Tesla" else (1 - 0.015*(gas_-3.5)) if brand=="Ford" else 1.0
        rv_   = base_ * mil_p * infl_ * int_ * gas_e * np.random.uniform(0.95,1.05)
        
        records.append({
            "make":brand, "model":mdl_, "msrp":msrp_, "age_years":age_,
            "mileage":mil_, "cpi_inflation":cpi_, "interest_rate":rate_,
            "gas_price":gas_, "residual_value":max(rv_,1000)
        })
        
    df = pd.DataFrame(records)
    df["make"]  = df["make"].astype("category")
    df["model"] = df["model"].astype("category")
    FEATURES = ["make","model","msrp","age_years","mileage","cpi_inflation","interest_rate","gas_price"]
    X, y = df[FEATURES], df["residual_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mdl = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.8,
                       tree_method="hist", enable_categorical=True, random_state=42)
    mdl.fit(X_train, y_train)
    return mdl

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "residual_model.json")
    try:
        mdl = XGBRegressor()
        mdl.load_model(model_path)
        return mdl
    except Exception:
        return _train_fresh_model()

model = load_model()

# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------
st.sidebar.markdown("### [ SYSTEM_CONFIG ]")
make = st.sidebar.selectbox("VEHICLE_MAKE", list(BRAND_MODELS.keys()))
model_name = st.sidebar.selectbox("VEHICLE_MODEL", BRAND_MODELS[make])
msrp = st.sidebar.number_input("MSRP ($)", value=DEFAULT_MSRP[make][model_name], step=1000)

age = st.sidebar.slider("CURRENT_AGE (YRS)", 0.0, 12.0, 3.0, 0.5)
mileage = st.sidebar.slider("CURRENT_MILEAGE", 0, 200_000, int(age * 12_000) if age > 0 else 100, 1_000)

st.sidebar.markdown("### [ MACRO_SHOCKS ]")
inflation = st.sidebar.slider("CPI_INFLATION (%)", 1.0, 12.0, 3.0, 0.1)
gas_price = st.sidebar.slider("GAS_PRICE ($/GAL)", 2.0, 8.0, 3.50, 0.10)
interest_rate = st.sidebar.slider("INTEREST_RATE (%)", 2.0, 12.0, 4.5, 0.1)

st.sidebar.markdown("### [ LOAN_PARAMETERS ]")
down_payment = st.sidebar.number_input("DOWN_PAYMENT ($)", value=int(msrp*0.1), step=500)
loan_term = st.sidebar.selectbox("LOAN_TERM (MONTHS)", [36, 48, 60, 72, 84], index=2)
loan_apr = st.sidebar.slider("LOAN_APR (%)", 1.0, 15.0, 6.5, 0.1)

# ---------------------------------------------------------------------------
# Core Prediction Logic
# ---------------------------------------------------------------------------
def predict_value(m_make, m_mod, m_msrp, m_age, m_mil, m_cpi, m_rate, m_gas):
    row = pd.DataFrame([{
        "make": m_make, "model": m_mod, "msrp": m_msrp, 
        "age_years": m_age, "mileage": m_mil, 
        "cpi_inflation": m_cpi, "interest_rate": m_rate, "gas_price": m_gas
    }])
    row["make"] = row["make"].astype("category")
    row["model"] = row["model"].astype("category")
    return float(max(model.predict(row)[0], 0))

current_value = predict_value(make, model_name, msrp, age, mileage, inflation, interest_rate, gas_price)
baseline_value = predict_value(make, model_name, msrp, age, mileage, BASELINE_MACRO["cpi_inflation"], BASELINE_MACRO["interest_rate"], BASELINE_MACRO["gas_price"])
macro_delta = current_value - baseline_value

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("AUTO_RESIDUAL_FORECASTER_V2")
st.markdown("**WARNING: RAW DATA PROJECTIONS ENABLED. BRUTALIST MODE ACTIVE.**")

tab1, tab2, tab3 = st.tabs(["// CORE_DASHBOARD", "// STRESS_MATRIX", "// EQUITY_ANALYSIS"])

# ==========================================
# TAB 1: CORE DASHBOARD
# ==========================================
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("EST_RESIDUAL_VALUE", f"${current_value:,.0f}", f"{(current_value/msrp*100):.1f}% RETENTION")
    col2.metric("MACRO_SHOCK_IMPACT", f"${macro_delta:+,.0f}", f"{(macro_delta/baseline_value*100 if baseline_value else 0):+.1f}% VS BASELINE", delta_color="normal")
    col3.metric("DEPRECIATION_COST", f"${(msrp - current_value):,.0f}", f"${((msrp - current_value)/(age*12) if age > 0 else 0):,.0f} / MONTH", delta_color="inverse")

    st.markdown("---")
    
    # 5-Year Forward Projection
    future_years = np.arange(0, 5.5, 0.5)
    labels = [f"Y+{y:.1f}" for y in future_years]
    
    val_shock, val_base = [], []
    for y in future_years:
        val_shock.append(predict_value(make, model_name, msrp, age+y, mileage+(y*12000), inflation, interest_rate, gas_price))
        val_base.append(predict_value(make, model_name, msrp, age+y, mileage+(y*12000), BASELINE_MACRO["cpi_inflation"], BASELINE_MACRO["interest_rate"], BASELINE_MACRO["gas_price"]))

    fig = go.Figure()
    # Baseline
    fig.add_trace(go.Scatter(x=labels, y=val_base, mode="lines+markers", name="BASELINE_MACRO", line=dict(color="#000000", width=4, dash="dot"), marker=dict(size=10, symbol="square")))
    # Shock
    fig.add_trace(go.Scatter(x=labels, y=val_shock, mode="lines+markers", name="SHOCK_SCENARIO", line=dict(color="#FF00FF", width=6), marker=dict(size=14, symbol="cross", line=dict(color="#000", width=2))))
    
    fig.update_layout(**get_brutalist_plotly_layout("5_YEAR_DEPRECIATION_TRAJECTORY"))
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: STRESS MATRIX
# ==========================================
with tab2:
    st.markdown("### SYSTEM_STRESS_TEST: INFLATION vs GAS PRICE")
    st.caption("Matrix displays projected residual value assuming holding current age/mileage constant.")
    
    inf_range = [1.0, 3.0, 5.0, 7.0, 9.0]
    gas_range = [2.0, 3.5, 5.0, 6.5, 8.0]
    
    matrix_data = []
    for i in inf_range:
        row_data = []
        for g in gas_range:
            v = predict_value(make, model_name, msrp, age, mileage, i, interest_rate, g)
            row_data.append(v)
        matrix_data.append(row_data)
        
    fig2 = go.Figure(data=go.Heatmap(
        z=matrix_data,
        x=[f"GAS_${g}" for g in gas_range],
        y=[f"INF_{i}%" for i in inf_range],
        colorscale=["#000000", "#FF00FF", "#00FFFF", "#E6FF00"], # Brutalist gradient
        text=[[f"${v:,.0f}" for v in row] for row in matrix_data],
        texttemplate="%{text}",
        hoverinfo="skip"
    ))
    fig2.update_layout(**get_brutalist_plotly_layout("SCENARIO_HEATMAP"))
    fig2.update_layout(margin=dict(l=80, r=20, t=60, b=50))
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 3: EQUITY ANALYSIS
# ==========================================
with tab3:
    st.markdown("### LOAN_AMORTIZATION_VS_DEPRECIATION")
    
    principal = msrp - down_payment
    r = (loan_apr / 100) / 12
    n = loan_term
    
    if principal > 0 and r > 0:
        monthly_payment = principal * (r * (1 + r)**n) / ((1 + r)**n - 1)
        
        months = np.arange(0, n+1, 6)
        balances = []
        car_values = []
        
        for m in months:
            # Remaining balance formula
            bal = principal * ((1+r)**n - (1+r)**m) / ((1+r)**n - 1)
            balances.append(max(bal, 0))
            
            # Depreciated value at month m (assuming starting from age=0 for the loan analysis)
            future_age = m / 12.0
            future_mil = future_age * 12000
            val = predict_value(make, model_name, msrp, future_age, future_mil, inflation, interest_rate, gas_price)
            car_values.append(val)
            
        fig3 = go.Figure()
        # Area between curves (Equity)
        equity = [cv - bal for cv, bal in zip(car_values, balances)]
        
        fig3.add_trace(go.Scatter(x=months, y=car_values, mode="lines", name="CAR_VALUE", line=dict(color="#00FFFF", width=5)))
        fig3.add_trace(go.Scatter(x=months, y=balances, mode="lines", name="LOAN_BALANCE", line=dict(color="#FF0000", width=5)))
        
        fig3.update_layout(**get_brutalist_plotly_layout("EQUITY_CURVE_ANALYSIS"))
        fig3.update_xaxes(title="MONTHS_SINCE_PURCHASE")
        fig3.update_yaxes(tickprefix="$")
        st.plotly_chart(fig3, use_container_width=True)
        
        underwater_months = [months[i] for i, eq in enumerate(equity) if eq < 0]
        if underwater_months:
            st.error(f"⚠️ DANGER: YOU ARE PROJECTED TO BE 'UNDERWATER' (OWE MORE THAN THE CAR IS WORTH) FOR SOME PORTION OF THIS LOAN.")
        else:
            st.success("✅ POSITIVE EQUITY RETAINED THROUGHOUT ENTIRE LOAN CYCLE.")
    else:
        st.warning("PLEASE ENTER VALID LOAN PARAMETERS IN SIDEBAR TO GENERATE EQUITY ANALYSIS.")

# ---------------------------------------------------------------------------
# Data Export
# ---------------------------------------------------------------------------
st.markdown("---")
export_df = pd.DataFrame({
    "YEAR_OFFSET": future_years,
    "AGE": projected_ages,
    "MILEAGE": projected_miles,
    "SHOCK_VALUE": val_shock,
    "BASELINE_VALUE": val_base
})
csv = export_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ DOWNLOAD_RAW_DATA.CSV",
    data=csv,
    file_name='auto_residual_projection.csv',
    mime='text/csv',
)
