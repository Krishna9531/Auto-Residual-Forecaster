# 🚗 Dynamic Residual Value Forecaster

Predict vehicle residual values under different macroeconomic scenarios using XGBoost and an interactive Streamlit dashboard.

## Features

- **XGBoost Regressor** trained on 15,000 simulated vehicle transactions across 5 brands (Toyota, Honda, Ford, BMW, Tesla)
- **Macroeconomic features**: CPI inflation, interest rates, gas prices
- **Interactive Streamlit dashboard** with:
  - Vehicle profile selectors (Make, Model, Age, Mileage)
  - Macro-shock sliders (Inflation, Gas Price, Interest Rate)
  - KPI metric cards (Estimated Value, Macro Impact, Confidence Interval)
  - 4-year forward depreciation curve (Plotly)

## Quick Start

```bash
# 1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train_model.py

# 4. Launch the dashboard
streamlit run app.py
```

## Project Structure

```
Auto Residual Forecaster/
├── data/               # Raw / processed datasets
├── models/             # Trained model artefacts
│   └── residual_model.json
├── notebooks/          # Exploratory notebooks
├── train_model.py      # Data simulation + model training pipeline
├── app.py              # Streamlit dashboard
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Model Details

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost (hist, enable_categorical) |
| Training samples | 12,000 |
| Test samples | 3,000 |
| Target MAPE | < 5 % |

## License

MIT
