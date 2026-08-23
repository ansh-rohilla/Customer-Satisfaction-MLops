import requests
import streamlit as st
from datetime import datetime


# ============================================================
# Configuration
# ============================================================

MLFLOW_ENDPOINT = "http://127.0.0.1:8002/invocations"

MODEL_NAME = "Customer-Satisfaction-Model"
MODEL_VERSION = "Champion"
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


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Customer Satisfaction AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Session State
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       Global
    ------------------------------------------------------- */

    html, body, [class*="css"] {
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            Roboto,
            Helvetica,
            Arial,
            sans-serif;
    }

    .stApp {
        background: #f6f8fc;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Remove excessive Streamlit top space */
    header {
        background: transparent !important;
    }

    /* -------------------------------------------------------
       Sidebar
    ------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    .sidebar-brand {
        padding: 0.5rem 0.25rem 1.8rem 0.25rem;
        border-bottom: 1px solid #273244;
        margin-bottom: 1.8rem;
    }

    .sidebar-brand-title {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .sidebar-brand-subtitle {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 0.35rem;
    }

    .sidebar-section {
        color: #64748b;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .sidebar-item {
        color: #cbd5e1;
        font-size: 0.84rem;
        padding: 0.55rem 0;
    }

    .sidebar-item strong {
        color: #ffffff;
        font-weight: 600;
    }

    .sidebar-status {
        background: #172033;
        border: 1px solid #26344a;
        border-radius: 10px;
        padding: 0.9rem;
        margin-top: 0.5rem;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 7px;
    }

    .status-online {
        color: #86efac;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .endpoint {
        color: #94a3b8;
        font-size: 0.68rem;
        word-break: break-all;
        margin-top: 0.45rem;
        line-height: 1.5;
    }

    /* -------------------------------------------------------
       Hero
    ------------------------------------------------------- */

    .hero {
        background:
            linear-gradient(
                135deg,
                #111827 0%,
                #172554 55%,
                #312e81 100%
            );
        border-radius: 22px;
        padding: 3rem 3.2rem;
        margin-bottom: 1.6rem;
        box-shadow:
            0 20px 50px rgba(15, 23, 42, 0.15);
    }

    .hero-label {
        display: inline-block;
        color: #a5b4fc;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }

    .hero-title {
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3.4rem);
        font-weight: 750;
        letter-spacing: -0.045em;
        line-height: 1.08;
        margin: 0;
    }

    .hero-description {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 800px;
        margin-top: 1rem;
        margin-bottom: 0;
    }

    /* -------------------------------------------------------
       Metric Cards
    ------------------------------------------------------- */

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 1.25rem 1.35rem;
        min-height: 110px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
    }

    .metric-label {
        color: #64748b;
        font-size: 0.67rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.18rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    .metric-description {
        color: #94a3b8;
        font-size: 0.72rem;
        margin-top: 0.25rem;
    }

    /* -------------------------------------------------------
       Section Headers
    ------------------------------------------------------- */

    .section-header {
        margin-top: 2rem;
        margin-bottom: 0.9rem;
    }

    .section-title {
        color: #0f172a;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.025em;
    }

    .section-description {
        color: #64748b;
        font-size: 0.84rem;
        margin-top: 0.25rem;
    }

    /* -------------------------------------------------------
       Input Cards
    ------------------------------------------------------- */

    .input-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.035);
    }

    .input-card-title {
        color: #1e293b;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .input-card-description {
        color: #94a3b8;
        font-size: 0.74rem;
        margin-bottom: 1rem;
    }

    /* Streamlit input labels */
    label {
        color: #334155 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }

    /* Number inputs */
    div[data-testid="stNumberInput"] input {
        background: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid #dbe3ee !important;
        border-radius: 9px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        min-height: 42px !important;
    }

    div[data-testid="stNumberInput"] input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.12) !important;
    }

    /* -------------------------------------------------------
       Prediction Button
    ------------------------------------------------------- */

    div.stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 11px;
        border: none;
        background: linear-gradient(
            135deg,
            #4f46e5,
            #6366f1
        );
        color: #ffffff;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.2);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(
            135deg,
            #4338ca,
            #4f46e5
        );
        transform: translateY(-1px);
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.27);
    }

    /* -------------------------------------------------------
       Prediction Result
    ------------------------------------------------------- */

    .prediction-container {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e293b 100%
        );
        border-radius: 20px;
        padding: 2.3rem;
        margin-top: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
    }

    .prediction-label {
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.13em;
    }

    .prediction-score {
        color: #ffffff;
        font-size: 4.4rem;
        font-weight: 800;
        letter-spacing: -0.06em;
        line-height: 1;
        margin-top: 0.7rem;
    }

    .prediction-scale {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.45rem;
    }

    .prediction-status {
        display: inline-block;
        margin-top: 1.2rem;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(129, 140, 248, 0.25);
        color: #c7d2fe;
        font-size: 0.82rem;
        font-weight: 700;
    }

    /* -------------------------------------------------------
       Progress
    ------------------------------------------------------- */

    .progress-wrapper {
        margin-top: 1.7rem;
    }

    .progress-background {
        height: 9px;
        width: 100%;
        background: #334155;
        border-radius: 20px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(
            90deg,
            #6366f1,
            #818cf8
        );
        border-radius: 20px;
    }

    .progress-labels {
        display: flex;
        justify-content: space-between;
        color: #64748b;
        font-size: 0.68rem;
        margin-top: 0.45rem;
    }

    /* -------------------------------------------------------
       Info Cards
    ------------------------------------------------------- */

    .info-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin-top: 1rem;
    }

    .info-title {
        color: #334155;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .info-value {
        color: #0f172a;
        font-size: 0.95rem;
        font-weight: 650;
        margin-top: 0.35rem;
    }

    /* -------------------------------------------------------
       History
    ------------------------------------------------------- */

    .history-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 1rem 1.2rem;
        margin-top: 0.8rem;
    }

    .history-score {
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 750;
    }

    .history-time {
        color: #94a3b8;
        font-size: 0.72rem;
    }

    /* -------------------------------------------------------
       Footer
    ------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.72rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
    }

    /* -------------------------------------------------------
       Hide Streamlit branding
    ------------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">Customer AI</div>
            <div class="sidebar-brand-subtitle">
                MLOps Prediction Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-item"><strong>Algorithm</strong></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-item">LightGBM Regressor</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-item"><strong>Task</strong></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-item">Customer Satisfaction Regression</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-item"><strong>Output Range</strong></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-item">1.0 — 5.0</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">MLOps Stack</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-item">
            ZenML — Pipeline orchestration
        </div>

        <div class="sidebar-item">
            MLflow — Experiment tracking
        </div>

        <div class="sidebar-item">
            MLflow — Model serving
        </div>

        <div class="sidebar-item">
            Streamlit — User interface
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Model Server</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-status">
            <div class="status-online">
                <span class="status-dot"></span>
                MLflow server configured
            </div>

            <div class="endpoint">
                {MLFLOW_ENDPOINT}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            margin-top: 2.5rem;
            color: #64748b;
            font-size: 0.68rem;
            line-height: 1.6;
        ">
            Customer Satisfaction MLOps<br>
            Built with ZenML and MLflow
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Hero Section
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-label">
            AI-POWERED CUSTOMER ANALYTICS
        </div>

        <h1 class="hero-title">
            Customer Satisfaction Predictor
        </h1>

        <p class="hero-description">
            Predict the expected customer review score using a
            production-oriented machine learning pipeline powered
            by ZenML, MLflow, and LightGBM.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model Overview Cards
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">LightGBM</div>
            <div class="metric-description">
                Gradient boosting regression
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Features</div>
            <div class="metric-value">12</div>
            <div class="metric-description">
                Input variables
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Framework</div>
            <div class="metric-value">ZenML</div>
            <div class="metric-description">
                Pipeline orchestration
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Serving</div>
            <div class="metric-value">MLflow</div>
            <div class="metric-description">
                Model deployment
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Main Input Section
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <div class="section-title">
            Order and Product Information
        </div>
        <div class="section-description">
            Enter the characteristics of the order to estimate
            customer satisfaction.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Payment Information
# ============================================================

st.markdown(
    """
    <div class="input-card">
        <div class="input-card-title">
            Payment and Order Details
        </div>
        <div class="input-card-description">
            Enter payment-related information for the order.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    payment_sequential = st.number_input(
        "Payment Sequential",
        min_value=0,
        value=1,
        step=1,
    )

with col2:
    payment_installments = st.number_input(
        "Payment Installments",
        min_value=0,
        value=1,
        step=1,
    )

with col3:
    payment_value = st.number_input(
        "Payment Value",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

with col4:
    price = st.number_input(
        "Product Price",
        min_value=0.0,
        value=80.0,
        step=10.0,
    )


# ============================================================
# Shipping Information
# ============================================================

st.markdown(
    """
    <div class="input-card">
        <div class="input-card-title">
            Shipping and Freight
        </div>
        <div class="input-card-description">
            Enter freight and physical product information.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    freight_value = st.number_input(
        "Freight Value",
        min_value=0.0,
        value=20.0,
        step=5.0,
    )

with col2:
    product_weight_g = st.number_input(
        "Product Weight (g)",
        min_value=0.0,
        value=500.0,
        step=50.0,
    )

with col3:
    product_length_cm = st.number_input(
        "Product Length (cm)",
        min_value=0.0,
        value=20.0,
        step=1.0,
    )

with col4:
    product_height_cm = st.number_input(
        "Product Height (cm)",
        min_value=0.0,
        value=10.0,
        step=1.0,
    )


# ============================================================
# Product Information
# ============================================================

st.markdown(
    """
    <div class="input-card">
        <div class="input-card-title">
            Product Characteristics
        </div>
        <div class="input-card-description">
            Enter product description and dimensional information.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    product_width_cm = st.number_input(
        "Product Width (cm)",
        min_value=0.0,
        value=15.0,
        step=1.0,
    )

with col2:
    product_name_lenght = st.number_input(
        "Product Name Length",
        min_value=0,
        value=40,
        step=1,
    )

with col3:
    product_description_lenght = st.number_input(
        "Description Length",
        min_value=0,
        value=200,
        step=10,
    )

with col4:
    product_photos_qty = st.number_input(
        "Product Photos",
        min_value=0,
        value=3,
        step=1,
    )


# ============================================================
# Prediction Button
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "Predict Customer Satisfaction",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Prediction
# ============================================================

if predict_button:

    input_data = [
        [
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
    ]

    payload = {
        "dataframe_split": {
            "columns": FEATURES,
            "data": input_data,
        }
    }

    with st.spinner("Running prediction through the MLflow model..."):

        try:

            response = requests.post(
                MLFLOW_ENDPOINT,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:

                result = response.json()

                prediction = float(
                    result["predictions"][0]
                )

                # Keep display value within review range.
                display_prediction = max(
                    1.0,
                    min(5.0, prediction)
                )

                # ------------------------------------------------
                # Determine interpretation
                # ------------------------------------------------

                if display_prediction >= 4.5:
                    status = "Excellent Satisfaction"
                    description = (
                        "The customer is likely to provide a "
                        "very positive review."
                    )

                elif display_prediction >= 4.0:
                    status = "Very Good Satisfaction"
                    description = (
                        "The customer is likely to be satisfied "
                        "with the purchase."
                    )

                elif display_prediction >= 3.0:
                    status = "Moderate Satisfaction"
                    description = (
                        "The prediction indicates moderate "
                        "satisfaction with room for improvement."
                    )

                else:
                    status = "Low Satisfaction"
                    description = (
                        "The order may have a higher probability "
                        "of receiving a negative review."
                    )

                # ------------------------------------------------
                # Save prediction to history
                # ------------------------------------------------

                st.session_state.prediction_history.insert(
                    0,
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "score": display_prediction,
                        "status": status,
                    },
                )

                # Keep only latest 10
                st.session_state.prediction_history = (
                    st.session_state.prediction_history[:10]
                )

                # ------------------------------------------------
                # Prediction Result
                # ------------------------------------------------

                st.markdown(
                    f"""
                    <div class="prediction-container">

                        <div class="prediction-label">
                            Predicted Customer Satisfaction
                        </div>

                        <div class="prediction-score">
                            {display_prediction:.2f}
                        </div>

                        <div class="prediction-scale">
                            out of 5.0
                        </div>

                        <div class="prediction-status">
                            {status}
                        </div>

                        <div class="progress-wrapper">

                            <div class="progress-background">
                                <div
                                    class="progress-fill"
                                    style="width:
                                    {display_prediction / 5 * 100}%"
                                ></div>
                            </div>

                            <div class="progress-labels">
                                <span>1.0</span>
                                <span>2.0</span>
                                <span>3.0</span>
                                <span>4.0</span>
                                <span>5.0</span>
                            </div>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------
                # Interpretation
                # ------------------------------------------------

                st.markdown(
                    f"""
                    <div class="info-card">

                        <div class="info-title">
                            Result Interpretation
                        </div>

                        <div class="info-value">
                            {description}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------
                # Prediction Details
                # ------------------------------------------------

                detail1, detail2, detail3 = st.columns(3)

                with detail1:
                    st.markdown(
                        f"""
                        <div class="info-card">
                            <div class="info-title">
                                Model
                            </div>
                            <div class="info-value">
                                {MODEL_ALGORITHM}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with detail2:
                    st.markdown(
                        f"""
                        <div class="info-card">
                            <div class="info-title">
                                Model Registry
                            </div>
                            <div class="info-value">
                                {MODEL_NAME}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with detail3:
                    st.markdown(
                        """
                        <div class="info-card">
                            <div class="info-title">
                                Serving
                            </div>
                            <div class="info-value">
                                MLflow Model Server
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ------------------------------------------------
                # API Response
                # ------------------------------------------------

                with st.expander("View MLflow API Response"):

                    st.json(result)

            else:

                st.error(
                    f"Prediction failed. "
                    f"HTTP Status: {response.status_code}"
                )

                st.code(response.text)

        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to the MLflow model server."
            )

            st.info(
                "Start the deployment first using:\n\n"
                "python run_deployment.py --config predict"
            )

        except requests.exceptions.Timeout:

            st.error(
                "The MLflow model server did not respond within "
                "the expected time."
            )

        except Exception as e:

            st.error(
                f"An unexpected error occurred: {str(e)}"
            )


# ============================================================
# Prediction History
# ============================================================

if st.session_state.prediction_history:

    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">
                Recent Predictions
            </div>
            <div class="section-description">
                Predictions generated during this session.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for item in st.session_state.prediction_history:

        score = item["score"]

        if score >= 4.5:
            indicator = "Excellent"
        elif score >= 4.0:
            indicator = "Very Good"
        elif score >= 3.0:
            indicator = "Moderate"
        else:
            indicator = "Low"

        st.markdown(
            f"""
            <div class="history-card">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                ">

                    <div>
                        <div class="history-score">
                            {score:.2f} / 5.0
                        </div>

                        <div class="history-time">
                            Prediction generated at
                            {item["time"]}
                        </div>
                    </div>

                    <div style="
                        color:#475569;
                        font-size:0.78rem;
                        font-weight:600;
                    ">
                        {indicator}
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MLOps Architecture
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <div class="section-title">
            MLOps Architecture
        </div>
        <div class="section-description">
            Production workflow used to generate the prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

arch1, arch2, arch3, arch4, arch5 = st.columns(5)

architecture = [
    ("Data", "Olist Dataset"),
    ("Pipeline", "ZenML"),
    ("Tracking", "MLflow"),
    ("Model", "LightGBM"),
    ("Serving", "MLflow API"),
]

for column, (title, value) in zip(
    [arch1, arch2, arch3, arch4, arch5],
    architecture,
):

    with column:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    {title}
                </div>

                <div class="metric-value">
                    {value}
                </div>

                <div class="metric-description">
                    MLOps component
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="footer">
        Customer Satisfaction MLOps Platform
        &nbsp;•&nbsp;
        ZenML
        &nbsp;•&nbsp;
        MLflow
        &nbsp;•&nbsp;
        LightGBM
        &nbsp;•&nbsp;
        Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)