import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ============================================================
# Page & Configuration Setup
# ============================================================

st.set_page_config(
    page_title="Customer Satisfaction AI | MLOps Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

MLFLOW_ENDPOINT = "http://127.0.0.1:8002/invocations"
MODEL_NAME = "Customer-Satisfaction-Model"
MODEL_ALIAS = "champion"
MODEL_ALGORITHM = "LightGBM Regressor"

FEATURES = [
    "payment_sequential",
    "payment_installments",
    "payment_value",
    "price",
    "freight_value",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]

PRESETS = {
    "VIP / High Satisfaction": {
        "payment_sequential": 1,
        "payment_installments": 1,
        "payment_value": 350.0,
        "price": 320.0,
        "freight_value": 12.0,
        "product_name_lenght": 58,
        "product_description_lenght": 1200,
        "product_photos_qty": 6,
        "product_weight_g": 450.0,
        "product_length_cm": 18.0,
        "product_height_cm": 12.0,
        "product_width_cm": 14.0,
    },
    "Express Standard": {
        "payment_sequential": 1,
        "payment_installments": 2,
        "payment_value": 110.0,
        "price": 90.0,
        "freight_value": 20.0,
        "product_name_lenght": 42,
        "product_description_lenght": 450,
        "product_photos_qty": 3,
        "product_weight_g": 800.0,
        "product_length_cm": 25.0,
        "product_height_cm": 15.0,
        "product_width_cm": 20.0,
    },
    "Heavy Freight & Slow": {
        "payment_sequential": 2,
        "payment_installments": 10,
        "payment_value": 180.0,
        "price": 70.0,
        "freight_value": 110.0,
        "product_name_lenght": 22,
        "product_description_lenght": 110,
        "product_photos_qty": 1,
        "product_weight_g": 12500.0,
        "product_length_cm": 70.0,
        "product_height_cm": 50.0,
        "product_width_cm": 45.0,
    },
    "High Risk / Poor Listing": {
        "payment_sequential": 3,
        "payment_installments": 12,
        "payment_value": 45.0,
        "price": 15.0,
        "freight_value": 30.0,
        "product_name_lenght": 15,
        "product_description_lenght": 60,
        "product_photos_qty": 1,
        "product_weight_g": 3200.0,
        "product_length_cm": 40.0,
        "product_height_cm": 30.0,
        "product_width_cm": 25.0,
    },
}

# Initialize Session State
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "use_mock_fallback" not in st.session_state:
    st.session_state.use_mock_fallback = False

# Populate default input values if not present
for preset_key, preset_val in PRESETS["Express Standard"].items():
    if preset_key not in st.session_state:
        st.session_state[preset_key] = preset_val


def apply_preset(preset_name: str):
    """Apply preset values to session state input keys."""
    if preset_name in PRESETS:
        for k, v in PRESETS[preset_name].items():
            st.session_state[k] = v


def mock_predict(feature_values: list) -> float:
    """Intelligent offline fallback predictor for demo purposes."""
    price = feature_values[3]
    freight = feature_values[4]
    desc_len = feature_values[6]
    photos = feature_values[7]
    weight = feature_values[8]

    freight_ratio = freight / max(1.0, (price + freight))
    score = 4.2

    if freight_ratio > 0.35:
        score -= 0.9
    elif freight_ratio < 0.15:
        score += 0.4

    if photos >= 4:
        score += 0.3
    elif photos == 1:
        score -= 0.3

    if desc_len > 800:
        score += 0.2
    elif desc_len < 100:
        score -= 0.4

    if weight > 8000:
        score -= 0.3

    return float(np.clip(score, 1.0, 5.0))


# ============================================================
# Inject Custom Modern CSS
# ============================================================

st.markdown(
    """
    <style>
    /* Global Styling & Reset */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }

    /* Top Padding & Block Constraints */
    .main .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .sidebar-brand {
        padding: 0.5rem 0.5rem 1.2rem 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.2rem;
    }

    .sidebar-brand-title {
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .sidebar-brand-sub {
        color: #9ca3af;
        font-size: 0.78rem;
        margin-top: 2px;
    }

    .sidebar-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 8px;
    }

    .badge-online {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .badge-mock {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    /* Hero Banner */
    .hero-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #172554 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 2.2rem 2.6rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.15;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        max-width: 850px;
        margin-top: 0.6rem;
        line-height: 1.6;
    }

    /* Card Panels */
    .glass-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }

    .glass-card-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .glass-card-sub {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-bottom: 1rem;
    }

    /* Metric Display Box */
    .metric-pill {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        text-align: center;
    }

    .metric-pill-val {
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #60a5fa;
    }

    .metric-pill-lbl {
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }

    /* Prediction Result Showcase */
    .result-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 20px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        position: relative;
    }

    .result-score {
        font-family: 'Outfit', sans-serif;
        font-size: 4.8rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.05em;
        margin: 0.5rem 0;
    }

    .score-emerald { color: #34d399; text-shadow: 0 0 30px rgba(52, 211, 153, 0.4); }
    .score-indigo  { color: #818cf8; text-shadow: 0 0 30px rgba(129, 140, 248, 0.4); }
    .score-amber   { color: #fbbf24; text-shadow: 0 0 30px rgba(251, 191, 36, 0.4); }
    .score-crimson { color: #f87171; text-shadow: 0 0 30px rgba(248, 113, 113, 0.4); }

    .result-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-top: 0.6rem;
    }

    .bg-emerald { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .bg-indigo  { background: rgba(129, 140, 248, 0.15); color: #818cf8; border: 1px solid rgba(129, 140, 248, 0.3); }
    .bg-amber   { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .bg-crimson { background: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }

    /* Custom Streamlit Input Overrides */
    div[data-testid="stNumberInput"] input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
    }

    div[data-testid="stNumberInput"] input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }

    div[data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(17, 24, 39, 0.8);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    div[data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }

    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
    }

    /* Footer */
    .footer-bar {
        text-align: center;
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar Component
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">
                Customer AI
            </div>
            <div class="sidebar-brand-sub">
                MLOps Prediction & Analytics Engine
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Execution Mode")
    mock_mode = st.toggle(
        "Offline Mock Mode (Demo)",
        value=st.session_state.use_mock_fallback,
        help="Enable offline predictions if local MLflow server is stopped.",
    )
    st.session_state.use_mock_fallback = mock_mode

    if mock_mode:
        st.markdown(
            '<div class="sidebar-badge badge-mock">● Mock Model Mode</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sidebar-badge badge-online">● MLflow Server Target</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Production MLOps Stack")
    st.markdown(
        """
        - **Pipeline Framework**: ZenML
        - **Experiment Tracking**: MLflow
        - **Model Registry**: Champion Alias
        - **Model Algorithm**: LightGBM
        - **REST Endpoint**: `:8002/invocations`
        """
    )

    st.markdown("---")
    st.markdown("### Dataset Reference")
    st.markdown(
        """
        - **Source**: Olist Brazilian E-Commerce
        - **Target**: Review Score (1.0 — 5.0)
        - **Features**: 12 Input Features
        """
    )

# ============================================================
# Hero Section
# ============================================================

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Customer Satisfaction Intelligence</div>
        <div class="hero-subtitle">
            Estimate e-commerce customer review scores in real-time using production-grade machine learning pipelines powered by ZenML, MLflow, and LightGBM.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Main Navigation Tabs
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Single Predictor",
        "Batch CSV Prediction",
        "MLOps System Health",
        "Session History Logs",
    ]
)

# ------------------------------------------------------------
# TAB 1: SINGLE PREDICTOR
# ------------------------------------------------------------

with tab1:

    # Presets Bar
    st.markdown("##### Quick Scenario Presets")
    preset_cols = st.columns(4)

    for i, (preset_name, _) in enumerate(PRESETS.items()):
        with preset_cols[i]:
            if st.button(
                preset_name, use_container_width=True, key=f"btn_preset_{i}"
            ):
                apply_preset(preset_name)
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form Layout
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:

        # Group 1: Order & Financials
        with st.container():
            st.markdown(
                """
                <div class="glass-card">
                    <div class="glass-card-header">Order & Payment Details</div>
                    <div class="glass-card-sub">Transaction values, payment methods, and installment structure.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                payment_sequential = st.number_input(
                    "Payment Sequences",
                    min_value=1,
                    max_value=10,
                    key="payment_sequential",
                )
                payment_value = st.number_input(
                    "Total Payment Value ($)",
                    min_value=0.0,
                    step=10.0,
                    key="payment_value",
                )

            with c2:
                payment_installments = st.number_input(
                    "Payment Installments",
                    min_value=1,
                    max_value=24,
                    key="payment_installments",
                )
                price = st.number_input(
                    "Product Price ($)",
                    min_value=0.0,
                    step=10.0,
                    key="price",
                )

        # Group 2: Freight & Physical Attributes
        with st.container():
            st.markdown(
                """
                <div class="glass-card">
                    <div class="glass-card-header">Shipping & Physical Characteristics</div>
                    <div class="glass-card-sub">Freight costs, weight, and dimensional measurements.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                freight_value = st.number_input(
                    "Freight Value ($)",
                    min_value=0.0,
                    step=5.0,
                    key="freight_value",
                )
                product_weight_g = st.number_input(
                    "Weight (g)",
                    min_value=0.0,
                    step=50.0,
                    key="product_weight_g",
                )

            with c2:
                product_length_cm = st.number_input(
                    "Length (cm)",
                    min_value=0.0,
                    step=1.0,
                    key="product_length_cm",
                )
                product_height_cm = st.number_input(
                    "Height (cm)",
                    min_value=0.0,
                    step=1.0,
                    key="product_height_cm",
                )

            with c3:
                product_width_cm = st.number_input(
                    "Width (cm)",
                    min_value=0.0,
                    step=1.0,
                    key="product_width_cm",
                )

        # Group 3: Listing Quality
        with st.container():
            st.markdown(
                """
                <div class="glass-card">
                    <div class="glass-card-header">Product Listing Quality</div>
                    <div class="glass-card-sub">Media richness and description completeness.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                product_name_lenght = st.number_input(
                    "Title Length",
                    min_value=1,
                    step=1,
                    key="product_name_lenght",
                )
            with c2:
                product_description_lenght = st.number_input(
                    "Description Length",
                    min_value=1,
                    step=20,
                    key="product_description_lenght",
                )
            with c3:
                product_photos_qty = st.number_input(
                    "Photo Count",
                    min_value=0,
                    step=1,
                    key="product_photos_qty",
                )

    with col_right:

        # Real-time Feature Analytics Panel
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-header">Derived Feature Analytics</div>
                <div class="glass-card-sub">Real-time calculated metrics sent to the model.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        freight_ratio = (freight_value / max(1.0, (price + freight_value))) * 100
        vol_cm3 = product_length_cm * product_height_cm * product_width_cm
        density = product_weight_g / max(1.0, vol_cm3)
        monthly = payment_value / max(1, payment_installments)

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"""
                <div class="metric-pill">
                    <div class="metric-pill-val">{freight_ratio:.1f}%</div>
                    <div class="metric-pill-lbl">Freight / Total Ratio</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="metric-pill">
                    <div class="metric-pill-val">${monthly:.2f}</div>
                    <div class="metric-pill-lbl">Monthly Installment</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-pill">
                    <div class="metric-pill-val">{vol_cm3:,.0f} cm³</div>
                    <div class="metric-pill-lbl">Product Volume</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="metric-pill">
                    <div class="metric-pill-val">{density:.2f} g/cm³</div>
                    <div class="metric-pill-lbl">Item Density</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        predict_btn = st.button(
            "Predict Customer Satisfaction",
            type="primary",
            use_container_width=True,
        )

        if predict_btn:

            raw_input = [
                payment_sequential,
                payment_installments,
                payment_value,
                price,
                freight_value,
                product_name_lenght,
                product_description_lenght,
                product_photos_qty,
                product_weight_g,
                product_length_cm,
                product_height_cm,
                product_width_cm,
            ]

            payload = {
                "dataframe_split": {
                    "columns": FEATURES,
                    "data": [raw_input],
                }
            }

            score_val = None
            is_mock_used = False
            latency_ms = 0

            with st.spinner("Processing prediction..."):
                start_t = time.time()

                if not st.session_state.use_mock_fallback:
                    try:
                        res = requests.post(
                            MLFLOW_ENDPOINT, json=payload, timeout=5
                        )
                        latency_ms = int((time.time() - start_t) * 1000)

                        if res.status_code == 200:
                            score_val = float(res.json()["predictions"][0])
                        else:
                            st.warning(
                                f"MLflow endpoint returned HTTP {res.status_code}. Using fallback mock predictor."
                            )
                            score_val = mock_predict(raw_input)
                            is_mock_used = True
                    except Exception:
                        st.info(
                            "MLflow endpoint unreachable. Switched to offline mock prediction."
                        )
                        score_val = mock_predict(raw_input)
                        is_mock_used = True
                else:
                    score_val = mock_predict(raw_input)
                    is_mock_used = True
                    latency_ms = int((time.time() - start_t) * 1000)

            display_score = float(np.clip(score_val, 1.0, 5.0))

            # Categorize sentiment
            if display_score >= 4.5:
                class_style = "score-emerald"
                badge_style = "bg-emerald"
                status_text = "Outstanding Satisfaction"
                desc_text = "The customer is highly likely to rate 5 stars. Delivery, pricing, and media listing parameters are optimized."
            elif display_score >= 4.0:
                class_style = "score-indigo"
                badge_style = "bg-indigo"
                status_text = "High Satisfaction"
                desc_text = "Positive review expected. Good overall balance between product pricing and shipping cost."
            elif display_score >= 3.0:
                class_style = "score-amber"
                badge_style = "bg-amber"
                status_text = "Moderate Satisfaction"
                desc_text = "Average score expected. Consider reducing freight costs or enhancing product photo count."
            else:
                class_style = "score-crimson"
                badge_style = "bg-crimson"
                status_text = "Low Satisfaction Risk"
                desc_text = "High probability of negative review. High freight-to-price ratio or sparse listing information detected."

            # Save to session history
            st.session_state.prediction_history.insert(
                0,
                {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "score": display_score,
                    "status": status_text,
                    "is_mock": is_mock_used,
                    "price": price,
                    "freight": freight_value,
                },
            )

            # Display Result Gauge Box
            st.markdown(
                f"""
                <div class="result-container">
                    <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;">
                        Predicted Customer Score
                    </div>
                    <div class="result-score {class_style}">
                        {display_score:.2f}
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.9rem;">out of 5.0</div>
                    <div class="result-badge {badge_style}">
                        {status_text}
                    </div>
                    <div style="margin-top: 1.2rem; font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                        {desc_text}
                    </div>
                    <div style="margin-top: 1rem; color: #64748b; font-size: 0.72rem;">
                        Latency: {latency_ms} ms &nbsp;•&nbsp; Engine: {"Mock Fallback" if is_mock_used else "MLflow Served Model"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------
# TAB 2: BATCH CSV PREDICTION
# ------------------------------------------------------------

with tab2:

    st.markdown(
        """
        <div class="glass-card">
            <div class="glass-card-header">Bulk Dataset Prediction</div>
            <div class="glass-card-sub">Upload a CSV file containing order features to predict scores for hundreds of transactions at once.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sample CSV Download Button
    sample_df = pd.DataFrame(
        [
            PRESETS["VIP / High Satisfaction"],
            PRESETS["Express Standard"],
            PRESETS["Heavy Freight & Slow"],
            PRESETS["High Risk / Poor Listing"],
        ]
    )

    csv_data = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Sample Batch Input CSV Template",
        data=csv_data,
        file_name="sample_customer_orders.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Order Dataset (.csv)", type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"Uploaded **{len(df_batch)}** rows.")

            missing_cols = [c for c in FEATURES if c not in df_batch.columns]

            if missing_cols:
                st.error(
                    f"Uploaded CSV is missing required model features: {missing_cols}"
                )
            else:
                if st.button("Run Batch Prediction", type="primary"):
                    batch_predictions = []

                    with st.spinner("Executing batch predictions..."):
                        for _, row in df_batch.iterrows():
                            row_vals = [row[f] for f in FEATURES]

                            if st.session_state.use_mock_fallback:
                                score = mock_predict(row_vals)
                            else:
                                try:
                                    res = requests.post(
                                        MLFLOW_ENDPOINT,
                                        json={
                                            "dataframe_split": {
                                                "columns": FEATURES,
                                                "data": [row_vals],
                                            }
                                        },
                                        timeout=3,
                                    )
                                    if res.status_code == 200:
                                        score = float(
                                            res.json()["predictions"][0]
                                        )
                                    else:
                                        score = mock_predict(row_vals)
                                except Exception:
                                    score = mock_predict(row_vals)

                            batch_predictions.append(
                                round(float(np.clip(score, 1.0, 5.0)), 2)
                            )

                    df_batch["Predicted_Review_Score"] = batch_predictions

                    st.success("Batch Prediction Complete!")

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric(
                            "Mean Score",
                            f"{np.mean(batch_predictions):.2f} / 5.0",
                        )
                    with m2:
                        st.metric("Min Score", f"{np.min(batch_predictions):.2f}")
                    with m3:
                        st.metric("Max Score", f"{np.max(batch_predictions):.2f}")

                    st.dataframe(df_batch, use_container_width=True)

                    out_csv = df_batch.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Scored Batch Results CSV",
                        data=out_csv,
                        file_name="customer_predictions_results.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")

# ------------------------------------------------------------
# TAB 3: MLOPS SYSTEM HEALTH
# ------------------------------------------------------------

with tab3:

    st.markdown(
        """
        <div class="glass-card">
            <div class="glass-card-header">MLOps Stack Architecture & Health Diagnostics</div>
            <div class="glass-card-sub">Real-time status of pipeline orchestrator, experiment tracker, and model endpoint.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    h1, h2, h3 = st.columns(3)

    with h1:
        st.markdown(
            """
            <div class="metric-pill">
                <div class="metric-pill-val">ZenML</div>
                <div class="metric-pill-lbl">Pipeline Orchestrator</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            """
            <div class="metric-pill">
                <div class="metric-pill-val">MLflow</div>
                <div class="metric-pill-lbl">Experiment Tracking & Registry</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h3:
        st.markdown(
            """
            <div class="metric-pill">
                <div class="metric-pill-val">LightGBM</div>
                <div class="metric-pill-lbl">Trained Regressor Model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Connection Diagnostic Button
    if st.button("Ping MLflow Server Endpoint"):
        start_ping = time.time()
        try:
            sample_payload = {
                "dataframe_split": {
                    "columns": FEATURES,
                    "data": [
                        [PRESETS["Express Standard"][f] for f in FEATURES]
                    ],
                }
            }
            res = requests.post(MLFLOW_ENDPOINT, json=sample_payload, timeout=4)
            ping_ms = int((time.time() - start_ping) * 1000)

            if res.status_code == 200:
                st.success(
                    f"Connection Successful! Response Status: HTTP {res.status_code} ({ping_ms} ms latency)"
                )
            else:
                st.warning(
                    f"Endpoint responded with HTTP Status {res.status_code} ({ping_ms} ms)"
                )
        except Exception as err:
            st.error(
                f"Connection Failed: MLflow model server is not reachable at {MLFLOW_ENDPOINT}. Details: {str(err)}"
            )

    st.markdown("### End-to-End Pipeline Workflow")
    st.code(
        """
[ Ingest Data ] -> [ Clean & Preprocess (Strategy Pattern) ] 
                 -> [ Multi-Model Optuna Fine Tuning ] 
                 -> [ Model Benchmark Evaluation ] 
                 -> [ Quality Gate Threshold Check (R² >= 0.08) ] 
                 -> [ Champion Tag Promotion in MLflow Registry ] 
                 -> [ Continuous REST Model Server Deployment ] 
                 -> [ Streamlit UI Consumption ]
    """,
        language="text",
    )

# ------------------------------------------------------------
# TAB 4: SESSION HISTORY LOGS
# ------------------------------------------------------------

with tab4:

    st.markdown(
        """
        <div class="glass-card">
            <div class="glass-card-header">Session Prediction History</div>
            <div class="glass-card-sub">Predictions recorded during your current active session.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.prediction_history:

        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)

    else:
        st.info("No predictions recorded in this session yet.")

# ============================================================
# Global Footer
# ============================================================

st.markdown(
    """
    <div class="footer-bar">
        Customer Satisfaction MLOps Platform &nbsp;•&nbsp; Powered by ZenML, MLflow, LightGBM & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)