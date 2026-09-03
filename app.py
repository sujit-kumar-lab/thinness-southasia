"""Thinness Prediction Decision Support System
A minimal, mobile-friendly Streamlit DSS for child and adolescent thinness prediction in South Asia
using a trained Mixed Effects Random Forest (MERF) model.

Model inputs: four determinants
  - Out-of-pocket expenditure
  - Urban population
  - Fertility rate
  - Unemployment

Random effects: country random intercept + country random time slope
Dataset: UNICEF + World Bank panel, 2000-2022, 8 South Asian countries
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import t

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

ROOT: Path = Path(__file__).resolve().parent
DATA_PATH: Path = ROOT / "data" / "panel_data_imputed.xlsx"
MODEL_PATH: Path = ROOT / "model" / "model_merf.pkl"
FEATURE_PATH: Path = ROOT / "model" / "feature_names.pkl"

# ============================================================================
# CONSTANTS
# ============================================================================

TIME_ORIGIN: int = 2000          # year zero-point for the random time slope
TARGET_COL: str = "thinness"     # prediction target column
DEFAULT_YEAR: int = 2030         # default projection year
MIN_YEAR: int = 2023
MAX_YEAR: int = 2030

# Friendly display labels for the 4 model input features.
FEATURE_LABELS: Dict[str, str] = {
    "oop_exp": "Out-of-pocket expenditure (%)",
    "urban_pop": "Urban population (%)",
    "fertility": "Fertility rate",
    "unemployment": "Unemployment (%)",
}


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="MERF-Based Decision Support System for Child and Adolescent Thinness",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS THEME
# ============================================================================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f5f8fb 0%, #eef3f7 100%);
        }

        .main .block-container {
            max-width: 1100px;
            padding: 0.8rem 0.85rem 1.1rem;
        }

        /* Header block */
        .app-header {
            background: linear-gradient(135deg, #14324a 0%, #1f5b7a 100%);
            color: #ffffff;
            border-radius: 10px;
            padding: 0.9rem 1rem 0.8rem;
            margin-bottom: 0.7rem;
            box-shadow: 0 4px 12px rgba(20, 50, 74, 0.12);
            text-align: center;
        }

        .app-header h1 {
            margin: 0;
            font-size: 1.42rem;
            line-height: 1.2;
            font-weight: 700;
            letter-spacing: -0.2px;
        }

        .app-header p {
            margin: 0.3rem 0 0;
            font-size: 0.76rem;
            opacity: 0.9;
        }

        /* Section labels */
        .section-title {
            color: #14324a;
            font-size: 0.96rem;
            font-weight: 700;
            margin: 0.75rem 0 0.4rem;
            padding-bottom: 0.22rem;
            border-bottom: 1px solid #d7e0e7;
        }

        /* Prediction panel */
        .pred-card {
            background: #ffffff;
            border: 1px solid #d7e0e7;
            border-left: 5px solid #1f6f8b;
            border-radius: 10px;
            height: 360px;
            box-sizing: border-box;
            padding: 1.1rem 1rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 3px 10px rgba(31, 91, 122, 0.07);
        }

        .pred-label {
            color: #63727f;
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .pred-value {
            color: #14324a;
            font-size: 3.15rem;
            line-height: 1;
            font-weight: 800;
            margin: 0.28rem 0 0.42rem;
        }

        .pred-unit {
            color: #63727f;
            font-size: 1.02rem;
            font-weight: 500;
        }

        .pred-ci {
            color: #45525d;
            font-size: 0.83rem;
            line-height: 1.45;
        }

        .meta-line {
            color: #6a7884;
            font-size: 0.76rem;
            margin-top: 0.6rem;
        }

        .method-box {
            background: #f0f5f8;
            border: 1px solid #d7e2e8;
            border-radius: 7px;
            padding: 0.58rem 0.68rem;
            font-size: 0.74rem;
            line-height: 1.45;
            color: #384853;
            margin-top: 0.6rem;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #d8e1e7;
        }

        section[data-testid="stSidebar"] .block-container {
            padding: 0.65rem 0.62rem 0.95rem;
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #14324a;
            font-size: 0.97rem;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] .stCaption {
            color: #6c7a86;
        }

        /* Metrics */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d7e0e7;
            border-radius: 7px;
            padding: 0.5rem 0.58rem;
            box-shadow: 0 2px 6px rgba(31, 91, 122, 0.04);
        }

        div[data-testid="stMetric"] label {
            color: #6a7884;
            font-size: 0.67rem;
        }

        div[data-testid="stMetricValue"] {
            color: #14324a;
            font-size: 1rem;
        }

        /* Institutional attribution badges */
        .source-strip {
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
            margin: 0.45rem 0 0.75rem;
        }

        .source-badge {
            background: #ffffff;
            border: 1px solid #d7e0e7;
            border-radius: 999px;
            padding: 0.28rem 0.65rem;
            color: #41515d;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        .source-badge.unicef {
            color: #0072a8;
        }

        .source-badge.worldbank {
            color: #1a4f7a;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #6a7884;
            font-size: 0.67rem;
            padding-top: 0.65rem;
            border-top: 1px solid #d8e1e7;
            margin-top: 0.8rem;
        }

        @media (max-width: 768px) {
            .main .block-container {
                padding: 0.5rem 0.45rem 0.85rem;
            }

            .app-header {
                padding: 0.72rem 0.65rem;
            }

            .app-header h1 {
                font-size: 1.1rem;
            }

            .app-header p {
                font-size: 0.68rem;
            }

            .pred-card {
                height: 300px;
            }

            .pred-value {
                font-size: 2.5rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# CACHED LOADERS
# ============================================================================

@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    """Load and lightly clean the panel dataset."""
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    return df


@st.cache_resource(show_spinner="Loading MERF model...")
def load_model() -> Tuple[object, List[str]]:
    """Load the trained 4-feature MERF model and normalize feature names."""
    with FEATURE_PATH.open("rb") as fh:
        raw_feature_names = pickle.load(fh)

    if hasattr(raw_feature_names, "tolist"):
        raw_feature_names = raw_feature_names.tolist()

    feature_names = [str(x).strip() for x in list(raw_feature_names)]
    model = joblib.load(MODEL_PATH)

    expected = {"oop_exp", "urban_pop", "fertility", "unemployment"}

    if len(feature_names) != 4 or set(feature_names) != expected:
        raise ValueError(
            "The loaded feature_names.pkl must contain exactly these four "
            "determinants: oop_exp, urban_pop, fertility, unemployment. "
            f"Loaded: {feature_names}"
        )

    return model, feature_names


@st.cache_data(show_spinner="Computing model performance...")
def compute_metrics(_model: object, df: pd.DataFrame, feature_names: List[str]) -> Dict[str, float]:
    """Compute in-sample performance metrics by running the MERF over the full observed panel."""
    X = df[feature_names].values
    Z = np.column_stack([np.ones(len(df)), df["year"].values - TIME_ORIGIN])
    clusters = df["country"]
    y_true = df[TARGET_COL].values
    y_pred = _model.predict(X, Z, clusters)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_latest_row(df: pd.DataFrame, country: str) -> pd.Series:
    """Return the most recent observed row for *country*."""
    country_df = df[df["country"] == country].sort_values("year")
    if country_df.empty:
        raise ValueError(f"No historical data for country: {country}")
    return country_df.iloc[-1]


def on_country_change(df: pd.DataFrame, feature_names: List[str]) -> None:
    """Callback fired when the user picks a new country. Seeds indicator widgets with latest values."""
    country = st.session_state["country_select"]
    latest = get_latest_row(df, country)
    for feat in feature_names:
        st.session_state[f"inp_{feat}"] = float(latest[feat])


def get_country_history(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Return the full chronological history for *country*."""
    return df[df["country"] == country].sort_values("year").reset_index(drop=True)


def prepare_merf_input(
    feature_names: List[str], country: str, year: int, user_values: Dict[str, float]
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Build the (X, Z, clusters) triple required by MERF.predict."""
    row = pd.DataFrame([{col: user_values[col] for col in feature_names}])
    X = row[feature_names]
    Z = np.array([[1.0, float(year - TIME_ORIGIN)]])
    clusters = pd.Series([country])
    return X, Z, clusters


def predict_thinness(
    model: object, feature_names: List[str], country: str, year: int, user_values: Dict[str, float]
) -> float:
    """Run MERF prediction and bound displayed prevalence to 0–100%."""
    X, Z, clusters = prepare_merf_input(feature_names, country, year, user_values)
    y_hat = model.predict(X, Z, clusters)
    return float(np.clip(y_hat[0], 0.0, 100.0))


def compute_approx_prediction_interval(
    model: object, feature_names: List[str], country: str, user_values: Dict[str, float], 
    df: pd.DataFrame, confidence: float = 0.95
) -> float:
    """Compute a prediction interval half-width combining tree variance and country-specific residual variance using Student's t distribution."""
    X = pd.DataFrame([{col: user_values[col] for col in feature_names}])
    
    tree_preds = np.array([tree.predict(X)[0] for tree in model.trained_fe_model.estimators_])
    tree_var = float(tree_preds.var(ddof=1)) if len(tree_preds) > 1 else 0.0

    country_df = df[df["country"] == country]
    if len(country_df) > len(feature_names) + 1:
        X_c = country_df[feature_names].values
        Z_c = np.column_stack([np.ones(len(country_df)), country_df["year"].values - TIME_ORIGIN])
        clusters_c = country_df["country"]
        y_true_c = country_df[TARGET_COL].values
        y_pred_c = model.predict(X_c, Z_c, clusters_c)
        residuals = y_true_c - y_pred_c
        country_resid_var = float(residuals.var(ddof=1))
        deg_freedom = len(country_df) - len(feature_names) - 1
    else:
        country_resid_var = float(((df[TARGET_COL] - model.predict(df[feature_names].values, np.column_stack([np.ones(len(df)), df["year"].values - TIME_ORIGIN]), df["country"])) ** 2).mean())
        deg_freedom = max(1, len(df) - len(feature_names) - 1)

    total_sd = float(np.sqrt(max(0.0, tree_var + country_resid_var)))
    t_crit = float(t.ppf(0.5 + confidence / 2.0, df=deg_freedom))
    return t_crit * total_sd


def build_trend_chart(
    history: pd.DataFrame, pred_year: int, pred_value: float, country: str, ci_half_width: float = 0.0
) -> go.Figure:
    """Build a Plotly line chart of historical thinness with the predicted point and 95% prediction interval."""
    fig = go.Figure()
    ci_lower = pred_value - ci_half_width
    ci_upper = pred_value + ci_half_width
    last_year = int(history["year"].iloc[-1])
    last_val = float(history[TARGET_COL].iloc[-1])

    fig.add_trace(
        go.Scatter(
            x=history["year"], y=history[TARGET_COL], mode="lines+markers", name="Historical",
            line=dict(color="#3413f1", width=2),
            marker=dict(size=7, color="#3413f1", line=dict(width=1, color="#ffffff")),
            hovertemplate="<b>%{x}</b><br>Thinness: %{y:.2f}%<extra></extra>",
        )
    )

    predicted_marker = dict(size=7, color="#d32f2f", symbol="circle", line=dict(width=1, color="#d41c1c"))
    predicted_trace = go.Scatter(
        x=[pred_year], y=[pred_value], mode="markers", name="Predicted", marker=predicted_marker,
        hovertemplate=f"<b>%{{x}}</b><br>Predicted: %{{y:.2f}}%<br>95% PI: {ci_lower:.2f}% – {ci_upper:.2f}%<extra></extra>",
    )
    if ci_half_width > 0:
        predicted_trace.error_y = dict(type="data", symmetric=True, array=[ci_half_width], color="#d32f2f", thickness=2, width=4)
    fig.add_trace(predicted_trace)

    if pred_year != last_year:
        fig.add_trace(
            go.Scatter(
                x=[last_year, pred_year], y=[last_val, pred_value], mode="lines", name="Projection",
                line=dict(color="#d32f2f", width=1.5, dash="dash"), hoverinfo="skip", showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(text=f"{country} : Trend & Projection", font=dict(size=18, color="#1a2a4a"), x=0.32),
        xaxis=dict(title="Year", gridcolor="#f1f3f5", tickfont=dict(size=14, color="#030303"), showline=True, linecolor="#dee2e6"),
        yaxis=dict(title="Thinness (%)", gridcolor="#f1f3f5", tickfont=dict(size=14, color="#030303"), showline=True, linecolor="#dee2e6"),
        legend=dict(orientation="h", y=1.07, x=0.5, xanchor="center", font=dict(size=14), bgcolor="rgba(255,255,255,0.9)"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", margin=dict(l=20, r=20, t=40, b=20), height=400,
    )
    return fig


def build_determinant_table(model: object, feature_names: List[str]) -> pd.DataFrame:
    """Return the four MERF fixed-effect feature importances."""
    importance = model.trained_fe_model.feature_importances_
    table = pd.DataFrame({
        "Determinant": [
            FEATURE_LABELS.get(f, f) for f in feature_names
        ],
        "Importance": importance,
    })
    return table.sort_values("Importance", ascending=False).reset_index(drop=True)


# ============================================================================
# DATA & MODEL LOADING
# ============================================================================

try:
    df = load_data()
    model, feature_names = load_model()
except FileNotFoundError as exc:
    st.error(f"Required file not found: {exc}. Ensure the data/ and model/ folders are in place.")
    st.stop()
except Exception as exc:
    st.error(f"Failed to load model or data: {exc}")
    st.stop()

countries: List[str] = sorted(df["country"].unique().tolist())


# ============================================================================
# PUBLICATION-READY INTERFACE
# ============================================================================

st.markdown(
    """
    <div class="app-header">
        <h1>MERF-Based Decision Support System for Child and Adolescent Thinness</h1>
        <p>Child and adolescent thinness prediction using four socioeconomic determinants • South Asia, 2000–2022</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="source-strip">
        <span class="source-badge unicef">UNICEF data</span>
        <span class="source-badge worldbank">World Bank indicators</span>
        <span class="source-badge">MERF • 4 determinants</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SIDEBAR (Custom Scenario Inputs Only)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Scenario specification")

    selected_country: str = st.selectbox(
        "Country",
        countries,
        index=0,
        key="country_select",
        on_change=on_country_change,
        args=(df, feature_names),
    )

    selected_year: int = st.number_input(
        "Year",
        min_value=2000,
        max_value=2030,
        value=DEFAULT_YEAR,
        step=1,
    )

    # Initialize default inputs from the latest observed row for the country if not set
    if f"init_{selected_country}" not in st.session_state:
        latest_row = get_latest_row(df, selected_country)
        for feat in feature_names:
            st.session_state[f"inp_{feat}"] = float(latest_row[feat])
        st.session_state[f"init_{selected_country}"] = True

    st.markdown("### Determinants")

    user_values: Dict[str, float] = {}
    input_cols = st.columns(2)

    for i, feat in enumerate(feature_names):
        col = input_cols[i % 2]
        latest_row = get_latest_row(df, selected_country)
        default_val = float(latest_row[feat])

        with col:
            user_values[feat] = float(
                st.number_input(
                    FEATURE_LABELS.get(feat, feat),
                    value=float(st.session_state.get(f"inp_{feat}", default_val)),
                    step=0.01,
                    format="%.2f",
                    key=f"inp_{feat}",
                )
            )

    st.markdown(
        '<div class="method-box">'
        '<b>Custom scenario mode:</b> adjust determinant values above to run real-time policy simulations for child and adolescent thinness.'
        '</div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------------------------
try:
    pred_value = predict_thinness(
        model, feature_names, selected_country, selected_year, user_values
    )
    ci_half = compute_approx_prediction_interval(
        model, feature_names, selected_country, user_values, df
    )
    ci_lower = pred_value - ci_half
    ci_upper = pred_value + ci_half
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

# ----------------------------------------------------------------------------
# MAIN RESULT
# ----------------------------------------------------------------------------
left, right = st.columns([0.9, 1.6], gap="medium")

with left:
    st.markdown(
        f"""
        <div class="pred-card">
            <div class="pred-label">Predicted thinness prevalence</div>
            <div class="pred-value">{pred_value:.2f}<span class="pred-unit">%</span></div>
            <div class="pred-ci">Approx. 95% prediction interval: {ci_lower:.2f}%–{ci_upper:.2f}%</div>
            <div class="meta-line">{selected_country} • {selected_year}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    history = get_country_history(df, selected_country)
    fig = build_trend_chart(
        history, selected_year, pred_value, selected_country, ci_half
    )
    fig.update_layout(
        title=dict(
            text=f"{selected_country}: observed trend and MERF projection",
            font=dict(size=15, color="#17324d"),
            x=0.5,
            xanchor="center",
        ),
        height=360,
        margin=dict(l=15, r=15, t=45, b=15),
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.5,
            xanchor="center",
            font=dict(size=10),
        ),
        xaxis=dict(title="Year", tickfont=dict(size=10)),
        yaxis=dict(title="Thinness (%)", tickfont=dict(size=10)),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"responsive": True, "displayModeBar": False},
    )

# ----------------------------------------------------------------------------
# DETERMINANT IMPORTANCE
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="section-title">Relative importance of the four determinants</div>',
    unsafe_allow_html=True,
)

importance_df = build_determinant_table(model, feature_names)

imp_cols = st.columns(4)
for i, (_, row) in enumerate(importance_df.iterrows()):
    with imp_cols[i]:
        st.metric(
            str(row["Determinant"]),
            f"{row['Importance']:.3f}",
        )

# ----------------------------------------------------------------------------
# VALIDATION SUMMARY
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="section-title">Temporal holdout validation</div>',
    unsafe_allow_html=True,
)

val_cols = st.columns(4)
validation_metrics = [
    ("RMSE", "0.3920"),
    ("MAE", "0.3183"),
    ("R²", "0.9941"),
    ("MAPE", "2.77%"),
]
for col, (label, value) in zip(val_cols, validation_metrics):
    with col:
        st.metric(label, value)

st.caption(
    "Evaluation: temporal holdout, training 2000–2017 and testing 2018–2022."
)

st.caption(
    "Projections beyond the observed period are model-based scenarios and should "
    "not be interpreted as observed estimates. The displayed interval is an "
    "approximate uncertainty interval based on tree variance and country-specific "
    "residual variance."
)

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        MERF-based DSS • Custom scenario analysis • UNICEF + World Bank panel • 8 South Asian countries
    </div>
    """,
    unsafe_allow_html=True,
)
