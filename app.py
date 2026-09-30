"""
app.py — Auto Residual Forecaster Dashboard (Medium Brutalism)
==============================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Page configuration & Medium Brutalist CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Auto Residual Forecaster",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Hide the top right toolbar, header, and footer entirely */
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    header {display: none !important;}
    footer {display: none !important;}
    #MainMenu {display: none !important;}

    /* Medium Brutalism: Clean but raw, stark contrast, sharp edges */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
        background-color: #FAFAFA !important; /* Crisp off-white */
        color: #000000 !important;
    }
    
    /* Headers with thick underlines */
    h1, h2, h3 {
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: -0.5px;
        border-bottom: 4px solid #000000;
        display: inline-block;
        padding-bottom: 5px;
        margin-bottom: 20px !important;
    }
    
    /* Metric Cards: Sharp, solid shadows */
    div[data-testid="metric-container"] {
        background: #FFFFFF;
        border: 3px solid #000000;
        padding: 15px 20px;
        box-shadow: 5px 5px 0px #000000; /* Solid offset shadow */
        border-radius: 0px !important; /* No rounded corners */
        transition: transform 0.1s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translate(-2px, -2px);
        box-shadow: 7px 7px 0px #0000FF; /* Stark blue interaction */
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-weight: 700;
        color: #000000 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 4px solid #000000 !important;
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 700 !important;
        text-transform: uppercase;
        border-radius: 0px !important;
        border: 2px solid transparent;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 2px solid #000000 !important;
    }
    
    /* Inputs: Harsh borders */
    .stSelectbox div[data-baseweb="select"] > div, 
    .stNumberInput div[data-baseweb="input"] {
        border-radius: 0px !important;
        border: 2px solid #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        border-radius: 0px !important;
        border: 2px solid #000000 !important;
    }
    
    /* Buttons */
    .stDownloadButton button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 3px solid #000000 !important;
        border-radius: 0px !important;
        box-shadow: 4px 4px 0px #FF3300 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
    }
    .stDownloadButton button:hover {
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0px #FF3300 !important;
    }
    
    /* Dividers */
    hr {
        border: none;
        border-top: 4px solid #000000;
        margin: 30px 0;
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

def get_medium_brutalist_layout(title=""):
    """Returns a medium brutalist dictionary for Plotly figure layouts."""
    return dict(
        title=dict(text=title.upper(), font=dict(family="Space Grotesk", size=18, color="#000")),
        font_family="Space Grotesk, sans-serif",
        font_color="#000000",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FAFAFA",
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(
            showgrid=True, gridcolor="#E0E0E0", gridwidth=1,
            zeroline=True, zerolinecolor="#000000", zerolinewidth=2,
            showline=True, linecolor="#000000", linewidth=3
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#E0E0E0", gridwidth=1,
            zeroline=True, zerolinecolor="#000000", zerolinewidth=2,
            showline=True, linecolor="#000000", linewidth=3
        ),
        hovermode="x unified"
    )

# ---------------------------------------------------------------------------
# Fallback Model Generator (Ultra-Fast)
# ---------------------------------------------------------------------------
def _train_fresh_model():
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
st.sidebar.markdown("### Vehicle Profile")
make = st.sidebar.selectbox("Make", list(BRAND_MODELS.keys()))
model_name = st.sidebar.selectbox("Model", BRAND_MODELS[make])
msrp = st.sidebar.number_input("Original MSRP ($)", value=DEFAULT_MSRP[make][model_name], step=1000)

age = st.sidebar.slider("Current Age (Years)", 0.0, 12.0, 3.0, 0.5)
mileage = st.sidebar.slider("Current Mileage", 0, 200_000, int(age * 12_000) if age > 0 else 100, 1_000)

st.sidebar.divider()
st.sidebar.markdown("### Macro Shocks")
inflation = st.sidebar.slider("CPI Inflation (%)", 1.0, 12.0, 3.0, 0.1)
gas_price = st.sidebar.slider("Gas Price ($/gal)", 2.0, 8.0, 3.50, 0.10)
interest_rate = st.sidebar.slider("Interest Rate (%)", 2.0, 12.0, 4.5, 0.1)

st.sidebar.divider()
st.sidebar.markdown("### Loan Params")
down_payment = st.sidebar.number_input("Down Payment ($)", value=int(msrp*0.1), step=500)
loan_term = st.sidebar.selectbox("Loan Term (Months)", [36, 48, 60, 72, 84], index=2)
loan_apr = st.sidebar.slider("Loan APR (%)", 1.0, 15.0, 6.5, 0.1)

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
st.title("Auto Residual Forecaster")

tab1, tab2, tab3 = st.tabs(["Dashboard", "Stress Matrix", "Equity Analysis"])

future_years = np.arange(0, 5.5, 0.5)
projected_ages = age + future_years
projected_miles = mileage + (future_years * 12_000)
labels = [f"Y+{y:.1f}" for y in future_years]

val_shock, val_base = [], []
for a, m in zip(projected_ages, projected_miles):
    val_shock.append(predict_value(make, model_name, msrp, a, m, inflation, interest_rate, gas_price))
    val_base.append(predict_value(make, model_name, msrp, a, m, BASELINE_MACRO["cpi_inflation"], BASELINE_MACRO["interest_rate"], BASELINE_MACRO["gas_price"]))


# ==========================================
# TAB 1: CORE DASHBOARD
# ==========================================
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Est. Residual Value", f"${current_value:,.0f}", f"{(current_value/msrp*100):.1f}% Retention")
    col2.metric("Macro Shock Impact", f"${macro_delta:+,.0f}", f"{(macro_delta/baseline_value*100 if baseline_value else 0):+.1f}% vs Baseline", delta_color="normal")
    col3.metric("Avg. Depreciation", f"${(msrp - current_value):,.0f}", f"${((msrp - current_value)/(age*12) if age > 0 else 0):,.0f} / month", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    
    fig = go.Figure()
    # Baseline
    fig.add_trace(go.Scatter(x=labels, y=val_base, mode="lines", name="Baseline", line=dict(color="#000000", width=3, dash="dot")))
    # Shock
    fig.add_trace(go.Scatter(x=labels, y=val_shock, mode="lines+markers", name="Adjusted", line=dict(color="#0000FF", width=5), marker=dict(size=10, color="#0000FF", symbol="square")))
    
    fig.update_layout(**get_medium_brutalist_layout("5-Year Depreciation Trajectory"))
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: STRESS MATRIX
# ==========================================
with tab2:
    st.markdown("### Stress Test Matrix")
    
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
        x=[f"${g}/gal" for g in gas_range],
        y=[f"{i}% CPI" for i in inf_range],
        colorscale=["#000000", "#FF3300", "#FFCC00", "#FFFFFF"], # Stark brutalist heatmap
        text=[[f"${v:,.0f}" for v in row] for row in matrix_data],
        texttemplate="%{text}",
        hoverinfo="skip",
        showscale=False
    ))
    fig2.update_layout(**get_medium_brutalist_layout())
    fig2.update_layout(margin=dict(l=60, r=20, t=20, b=40))
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 3: EQUITY ANALYSIS
# ==========================================
with tab3:
    st.markdown("### Loan Amortization vs. Depreciation")
    
    principal = msrp - down_payment
    r = (loan_apr / 100) / 12
    n = loan_term
    
    if principal > 0 and r > 0:
        monthly_payment = principal * (r * (1 + r)**n) / ((1 + r)**n - 1)
        
        months = np.arange(0, n+1, 6)
        balances = []
        car_values = []
        
        for m in months:
            bal = principal * ((1+r)**n - (1+r)**m) / ((1+r)**n - 1)
            balances.append(max(bal, 0))
            
            future_age = m / 12.0
            future_mil = future_age * 12000
            val = predict_value(make, model_name, msrp, future_age, future_mil, inflation, interest_rate, gas_price)
            car_values.append(val)
            
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(x=months, y=car_values, mode="lines", name="Vehicle Value", fill='tozeroy', fillcolor="rgba(0, 0, 255, 0.1)", line=dict(color="#0000FF", width=4)))
        fig3.add_trace(go.Scatter(x=months, y=balances, mode="lines", name="Loan Balance", line=dict(color="#FF3300", width=4)))
        
        fig3.update_layout(**get_medium_brutalist_layout("Equity Curve"))
        fig3.update_xaxes(title="Months Since Purchase")
        fig3.update_yaxes(tickprefix="$")
        st.plotly_chart(fig3, use_container_width=True)
        
        equity = [cv - bal for cv, bal in zip(car_values, balances)]
        underwater_months = [months[i] for i, eq in enumerate(equity) if eq < 0]
        
        if underwater_months:
            st.error(f"DANGER: You are projected to be 'underwater' on this loan.")
        else:
            st.success("SAFE: Positive equity retained throughout.")
    else:
        st.info("Enter valid loan parameters in sidebar.")

# ---------------------------------------------------------------------------
# Data Export
# ---------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
export_df = pd.DataFrame({
    "Year Offset": future_years,
    "Projected Age": projected_ages,
    "Projected Mileage": projected_miles,
    "Adjusted Value": val_shock,
    "Baseline Value": val_base
})

colA, colB = st.columns([1, 4])
with colA:
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="DOWNLOAD CSV",
        data=csv,
        file_name='auto_residual_projection.csv',
        mime='text/csv',
    )
