"""
app.py — Auto Residual Forecaster Dashboard (Y2K / Neo-Brutalism Edition)
==========================================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Page configuration & Y2K Neo-Brutalist CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Auto Residual Forecaster",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Hide top right toolbar, header, and footer */
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    header {display: none !important;}
    footer {display: none !important;}
    #MainMenu {display: none !important;}

    /* Y2K Neo-Brutalism: Lime Green + Electric Blue */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800;900&display=swap');
    
    :root {
        --bg-lime: #D4FF00;
        --y2k-blue: #0A00FF;
        --border-thick: 4px solid var(--y2k-blue);
        --border-dashed: 4px dashed var(--y2k-blue);
    }
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
        background-color: var(--bg-lime) !important;
        color: var(--y2k-blue) !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 900 !important;
        color: var(--y2k-blue) !important;
        text-transform: uppercase;
        letter-spacing: -1px;
    }
    
    h1 {
        border: var(--border-dashed);
        border-radius: 50px;
        padding: 15px 40px;
        background: #FFFFFF;
        display: inline-block;
        box-shadow: 6px 6px 0px var(--y2k-blue);
    }
    
    /* Metric Cards - Y2K Style */
    div[data-testid="metric-container"] {
        background: #FFFFFF;
        border: var(--border-thick);
        padding: 20px 30px;
        border-radius: 40px !important; /* Pill shapes */
        box-shadow: 6px 6px 0px var(--y2k-blue);
        transition: all 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0px var(--y2k-blue);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-weight: 900;
        color: var(--y2k-blue) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 800 !important;
        font-size: 1rem !important;
        text-transform: uppercase;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: var(--border-thick) !important;
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 900 !important;
        color: var(--y2k-blue) !important;
        text-transform: uppercase;
        border: var(--border-thick) !important;
        border-radius: 40px !important;
        margin-right: 15px !important;
        background: #FFFFFF;
        padding: 5px 25px !important;
        box-shadow: 4px 4px 0px var(--y2k-blue);
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--y2k-blue) !important;
        color: #FFFFFF !important;
    }
    
    /* Inputs */
    .stSelectbox div[data-baseweb="select"] > div, 
    .stNumberInput div[data-baseweb="input"] {
        border-radius: 30px !important;
        border: var(--border-thick) !important;
        background-color: #FFFFFF !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: var(--y2k-blue) !important; 
    }
    
    /* Download Button */
    .stDownloadButton button {
        background-color: #FFFFFF !important;
        color: var(--y2k-blue) !important;
        border: var(--border-thick) !important;
        border-radius: 50px !important;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        padding: 10px 40px !important;
        box-shadow: 6px 6px 0px var(--y2k-blue);
        transition: all 0.2s;
    }
    .stDownloadButton button:hover {
        background-color: var(--y2k-blue) !important;
        color: #FFFFFF !important;
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0px var(--y2k-blue) !important;
    }
    
    /* Dividers */
    hr {
        border: none;
        border-top: var(--border-dashed) !important;
        margin: 3rem 0;
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

def get_y2k_layout(title=""):
    """Returns a highly stylized Y2K Neo-Brutalist dictionary for Plotly figure layouts."""
    return dict(
        title=dict(text=f"✦ {title.upper()} ✦", font=dict(family="Outfit", size=20, color="#0A00FF")),
        font_family="Outfit, sans-serif",
        font_color="#0A00FF",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            showgrid=True, gridcolor="#0A00FF", gridwidth=2, griddash="dot",
            zeroline=True, zerolinecolor="#0A00FF", zerolinewidth=4,
            showline=True, linecolor="#0A00FF", linewidth=4, mirror=True,
            tickfont=dict(weight="bold")
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#0A00FF", gridwidth=2, griddash="dot",
            zeroline=True, zerolinecolor="#0A00FF", zerolinewidth=4,
            showline=True, linecolor="#0A00FF", linewidth=4, mirror=True,
            tickfont=dict(weight="bold")
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=15, font_family="Outfit", bordercolor="#0A00FF")
    )

# ---------------------------------------------------------------------------
# Fallback Model Generator
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
st.sidebar.markdown("### ⊚ VEHICLE PROFILE")
make = st.sidebar.selectbox("Make", list(BRAND_MODELS.keys()))
model_name = st.sidebar.selectbox("Model", BRAND_MODELS[make])
msrp = st.sidebar.number_input("Original MSRP ($)", value=DEFAULT_MSRP[make][model_name], step=1000)

age = st.sidebar.slider("Current Age (Years)", 0.0, 12.0, 3.0, 0.5)
mileage = st.sidebar.slider("Current Mileage", 0, 200_000, int(age * 12_000) if age > 0 else 100, 1_000)

st.sidebar.divider()
st.sidebar.markdown("### ⊚ MACRO CONDITIONS")
inflation = st.sidebar.slider("CPI Inflation (%)", 1.0, 12.0, 3.0, 0.1)
gas_price = st.sidebar.slider("Gas Price ($/gal)", 2.0, 8.0, 3.50, 0.10)
interest_rate = st.sidebar.slider("Interest Rate (%)", 2.0, 12.0, 4.5, 0.1)

st.sidebar.divider()
st.sidebar.markdown("### ⊚ LOAN PARAMETERS")
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
st.markdown("<h1><span style='font-family: monospace;'>[+]</span> RESIDUAL FORECASTER</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.3rem; font-weight: 600;'>✦ ADVANCED ANALYTICS FOR VEHICLE DEPRECIATION ✦</p><br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["( ⊚ OVERVIEW )", "( ⊞ SCENARIO MATRIX )", "( ⊟ EQUITY ANALYSIS )"])

future_years = np.arange(0, 5.5, 0.5)
projected_ages = age + future_years
projected_miles = mileage + (future_years * 12_000)
labels = [f"Y+{y:.1f}" for y in future_years]

val_shock, val_base = [], []
for a, m in zip(projected_ages, projected_miles):
    val_shock.append(predict_value(make, model_name, msrp, a, m, inflation, interest_rate, gas_price))
    val_base.append(predict_value(make, model_name, msrp, a, m, BASELINE_MACRO["cpi_inflation"], BASELINE_MACRO["interest_rate"], BASELINE_MACRO["gas_price"]))

# ==========================================
# TAB 1: OVERVIEW
# ==========================================
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("EST. RESIDUAL VALUE", f"${current_value:,.0f}", f"{(current_value/msrp*100):.1f}% Retention")
    col2.metric("MACRO SHOCK IMPACT", f"${macro_delta:+,.0f}", f"{(macro_delta/baseline_value*100 if baseline_value else 0):+.1f}% vs Baseline")
    col3.metric("AVG. DEPRECIATION", f"${(msrp - current_value):,.0f}", f"${((msrp - current_value)/(age*12) if age > 0 else 0):,.0f} / month")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Wrap the graph in a styled container
    st.markdown("<div style='border: 4px solid #0A00FF; border-radius: 40px; box-shadow: 6px 6px 0px #0A00FF; overflow: hidden;'>", unsafe_allow_html=True)
    fig = go.Figure()
    # Baseline
    fig.add_trace(go.Scatter(x=labels, y=val_base, mode="lines", name="Baseline", line=dict(color="#000000", width=4, dash="dash")))
    # Shock (Primary Accent)
    fig.add_trace(go.Scatter(x=labels, y=val_shock, mode="lines+markers", name="Adjusted", line=dict(color="#0A00FF", width=6), marker=dict(size=14, color="#FFFFFF", line=dict(color="#0A00FF", width=4))))
    
    fig.update_layout(**get_y2k_layout("5-Year Depreciation Trajectory"))
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: STRESS MATRIX
# ==========================================
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    
    inf_range = [1.0, 3.0, 5.0, 7.0, 9.0]
    gas_range = [2.0, 3.5, 5.0, 6.5, 8.0]
    
    matrix_data = []
    for i in inf_range:
        row_data = []
        for g in gas_range:
            v = predict_value(make, model_name, msrp, age, mileage, i, interest_rate, g)
            row_data.append(v)
        matrix_data.append(row_data)
        
    st.markdown("<div style='border: 4px solid #0A00FF; border-radius: 40px; box-shadow: 6px 6px 0px #0A00FF; padding: 20px; background: #FFFFFF;'>", unsafe_allow_html=True)
    fig2 = go.Figure(data=go.Heatmap(
        z=matrix_data,
        x=[f"${g}/gal" for g in gas_range],
        y=[f"{i}% CPI" for i in inf_range],
        colorscale=["#FFFFFF", "#0A00FF"], 
        text=[[f"${v:,.0f}" for v in row] for row in matrix_data],
        texttemplate="%{text}",
        hoverinfo="skip",
        showscale=False
    ))
    fig2.update_layout(**get_y2k_layout("Stress Test Matrix"))
    fig2.update_layout(margin=dict(l=60, r=40, t=60, b=40))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 3: EQUITY ANALYSIS
# ==========================================
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    
    principal = msrp - down_payment
    r = (loan_apr / 100) / 12
    n = loan_term
    
    if principal > 0 and r > 0:
        monthly_payment = principal * (r * (1 + r)**n) / ((1 + r)**n - 1)
        months = np.arange(0, n+1, 6)
        balances, car_values = [], []
        
        for m in months:
            bal = principal * ((1+r)**n - (1+r)**m) / ((1+r)**n - 1)
            balances.append(max(bal, 0))
            
            future_age = m / 12.0
            future_mil = future_age * 12000
            val = predict_value(make, model_name, msrp, future_age, future_mil, inflation, interest_rate, gas_price)
            car_values.append(val)
            
        st.markdown("<div style='border: 4px solid #0A00FF; border-radius: 40px; box-shadow: 6px 6px 0px #0A00FF; overflow: hidden;'>", unsafe_allow_html=True)
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(x=months, y=car_values, mode="lines+markers", name="Vehicle Value", fill='tozeroy', fillcolor="rgba(10, 0, 255, 0.1)", line=dict(color="#0A00FF", width=5), marker=dict(size=10, color="#FFFFFF", line=dict(color="#0A00FF", width=3))))
        fig3.add_trace(go.Scatter(x=months, y=balances, mode="lines+markers", name="Loan Balance", line=dict(color="#000000", width=5), marker=dict(size=10, color="#FFFFFF", line=dict(color="#000000", width=3))))
        
        fig3.update_layout(**get_y2k_layout("Equity Curve"))
        fig3.update_xaxes(title="Months Since Purchase")
        fig3.update_yaxes(tickprefix="$")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div><br>", unsafe_allow_html=True)
        
        equity = [cv - bal for cv, bal in zip(car_values, balances)]
        underwater_months = [months[i] for i, eq in enumerate(equity) if eq < 0]
        
        if underwater_months:
            st.warning("You are projected to have negative equity (owe more than the car is worth) during a portion of this loan.")
        else:
            st.success("You are projected to maintain positive equity throughout the entire loan cycle.")
    else:
        st.info("Enter valid loan parameters in the sidebar.")

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
        label="( downarrow DOWNLOAD CSV )",
        data=csv,
        file_name='auto_residual_projection.csv',
        mime='text/csv',
    )
