import os
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

MLFLOW_ENDPOINT = os.getenv(
    "MLFLOW_ENDPOINT",
    "http://127.0.0.1:8002/invocations",
)

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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Satisfaction AI",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(139, 92, 246, 0.08),
                transparent 30%
            ),
            #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */

    .hero {
        padding: 2.5rem 2.8rem;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                #111827 0%,
                #1e293b 50%,
                #312e81 100%
            );
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.15);
    }

    .hero-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.15);
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
    }

    .hero-subtitle {
        margin-top: 0.9rem;
        font-size: 1.05rem;
        color: #cbd5e1;
        max-width: 800px;
        line-height: 1.7;
    }


    /* --------------------------------------------------------
       STAT CARDS
    -------------------------------------------------------- */

    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.25rem;
        min-height: 125px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.05);
    }

    .stat-label {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stat-value {
        color: #0f172a;
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 0.45rem;
    }

    .stat-description {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 0.3rem;
    }


    /* --------------------------------------------------------
       SECTION HEADERS
    -------------------------------------------------------- */

    .section-header {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }

    .section-description {
        color: #64748b;
        font-size: 0.9rem;
    }


    /* --------------------------------------------------------
       INPUT CARDS
    -------------------------------------------------------- */

    .input-card {
        background: white;
        padding: 1.3rem;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }


    /* --------------------------------------------------------
       PREDICTION CARD
    -------------------------------------------------------- */

    .prediction-card {
        margin-top: 1.5rem;
        padding: 2.5rem;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                #312e81 0%,
                #4f46e5 45%,
                #7c3aed 100%
            );
        color: white;
        text-align: center;
        box-shadow:
            0 20px 50px rgba(79, 70, 229, 0.25);
    }

    .prediction-label {
        font-size: 0.95rem;
        color: #ddd6fe;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-value {
        font-size: 4.5rem;
        line-height: 1;
        font-weight: 850;
        margin: 1rem 0 0.7rem;
    }

    .prediction-scale {
        color: #ddd6fe;
        font-size: 0.9rem;
    }


    /* --------------------------------------------------------
       RESULT BADGE
    -------------------------------------------------------- */

    .result-badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.55rem 1rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.2);
        font-size: 0.9rem;
        font-weight: 650;
    }


    /* --------------------------------------------------------
       INFO BOX
    -------------------------------------------------------- */

    .info-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.4rem;
        margin-top: 1rem;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.04);
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .custom-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        padding: 2rem 0 1rem;
    }


    /* --------------------------------------------------------
       BUTTON
    -------------------------------------------------------- */

    .stButton > button {
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 700;
        font-size: 1rem;
    }


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⭐ Customer AI")

    st.caption("MLOps Prediction Platform")

    st.divider()

    st.markdown("### 🤖 Model")

    st.write("**Algorithm**")
    st.caption("LightGBM Regressor")

    st.write("**Task**")
    st.caption("Customer Satisfaction Regression")

    st.write("**Output Range**")
    st.caption("1.0 — 5.0 ⭐")

    st.divider()

    st.markdown("### ⚙️ MLOps Stack")

    st.caption("🧩 ZenML — Pipeline orchestration")
    st.caption("📊 MLflow — Experiment tracking")
    st.caption("🚀 MLflow — Model serving")
    st.caption("🎨 Streamlit — User interface")

    st.divider()

    st.markdown("### 🌐 Model Server")

    st.code(
        MLFLOW_ENDPOINT,
        language="text",
    )

    st.divider()

    st.caption(
        "Customer Satisfaction MLOps\n"
        "Built with ZenML + MLflow"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            ✨ AI-POWERED CUSTOMER ANALYTICS
        </div>

        <div class="hero-title">
            Customer Satisfaction Predictor
        </div>

        <div class="hero-subtitle">
            Predict the expected customer review score using a production-oriented
            machine learning pipeline powered by ZenML, MLflow and LightGBM.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL STATUS CARDS
# ============================================================

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-label">Model</div>
            <div class="stat-value">LightGBM</div>
            <div class="stat-description">Regression model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col2:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-label">Features</div>
            <div class="stat-value">12</div>
            <div class="stat-description">Input variables</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col3:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-label">Framework</div>
            <div class="stat-value">ZenML</div>
            <div class="stat-description">Pipeline orchestration</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col4:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-label">Serving</div>
            <div class="stat-value">MLflow</div>
            <div class="stat-description">Model deployment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <div class="section-title">📦 Order & Product Information</div>
        <div class="section-description">
            Enter the characteristics of the order to estimate customer satisfaction.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAYMENT & ORDER INFORMATION
# ============================================================

st.markdown(
    """
    <div class="input-card">
        <strong>💳 Payment & Order Details</strong>
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
        help="Sequential number of the payment method.",
    )

with col2:

    payment_installments = st.number_input(
        "Payment Installments",
        min_value=0,
        value=1,
        step=1,
        help="Number of installments used for payment.",
    )

with col3:

    payment_value = st.number_input(
        "Payment Value",
        min_value=0.0,
        value=100.0,
        step=10.0,
        help="Total payment value.",
    )

with col4:

    price = st.number_input(
        "Product Price",
        min_value=0.0,
        value=80.0,
        step=10.0,
        help="Price of the purchased product.",
    )


# ============================================================
# SHIPPING INFORMATION
# ============================================================

st.markdown(
    """
    <div class="input-card">
        <strong>🚚 Shipping & Freight</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:

    freight_value = st.number_input(
        "Freight Value",
        min_value=0.0,
        value=20.0,
        step=5.0,
        help="Shipping/freight cost.",
    )

with col2:

    product_weight_g = st.number_input(
        "Product Weight (g)",
        min_value=0.0,
        value=500.0,
        step=50.0,
        help="Product weight in grams.",
    )

with col3:

    product_photos_qty = st.number_input(
        "Product Photos",
        min_value=0,
        value=3,
        step=1,
        help="Number of photos available for the product.",
    )


# ============================================================
# PRODUCT INFORMATION
# ============================================================

st.markdown(
    """
    <div class="input-card">
        <strong>📦 Product Characteristics</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    product_name_lenght = st.number_input(
        "Product Name Length",
        min_value=0,
        value=40,
        step=1,
    )

with col2:

    product_description_lenght = st.number_input(
        "Description Length",
        min_value=0,
        value=200,
        step=10,
    )

with col3:

    product_length_cm = st.number_input(
        "Length (cm)",
        min_value=0.0,
        value=20.0,
        step=1.0,
    )

with col4:

    product_height_cm = st.number_input(
        "Height (cm)",
        min_value=0.0,
        value=10.0,
        step=1.0,
    )


col1, col2 = st.columns(2)

with col1:

    product_width_cm = st.number_input(
        "Width (cm)",
        min_value=0.0,
        value=15.0,
        step=1.0,
    )

with col2:

    st.markdown(
        """
        <div class="info-box">
            <strong>💡 Prediction Tip</strong><br><br>
            Product characteristics, payment behavior and freight costs
            can influence the expected customer satisfaction score.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮  Predict Customer Satisfaction",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PREDICTION
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

    with st.spinner("🤖 Running prediction through MLflow..."):

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

                # Keep score inside the normal review range.
                display_prediction = max(
                    1.0,
                    min(5.0, prediction),
                )

                # ====================================================
                # INTERPRETATION
                # ====================================================

                if display_prediction >= 4.5:

                    category = "Excellent Satisfaction"
                    emoji = "🌟"
                    message = (
                        "The customer is highly likely to have "
                        "a very positive experience."
                    )

                elif display_prediction >= 4.0:

                    category = "Very Good Satisfaction"
                    emoji = "😊"
                    message = (
                        "The customer is likely to be satisfied "
                        "with the purchase."
                    )

                elif display_prediction >= 3.0:

                    category = "Moderate Satisfaction"
                    emoji = "😐"
                    message = (
                        "The predicted experience is moderate. "
                        "There may be opportunities for improvement."
                    )

                else:

                    category = "Low Satisfaction"
                    emoji = "⚠️"
                    message = (
                        "The order may have a higher risk of "
                        "resulting in a negative customer review."
                    )


                # ====================================================
                # PREDICTION CARD
                # ====================================================

                st.markdown(
                    f"""
                    <div class="prediction-card">

                        <div class="prediction-label">
                            Predicted Customer Satisfaction
                        </div>

                        <div class="prediction-value">
                            ⭐ {display_prediction:.2f}
                        </div>

                        <div class="prediction-scale">
                            Expected review score out of 5
                        </div>

                        <div class="result-badge">
                            {emoji} {category}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                # ====================================================
                # RESULT DETAILS
                # ====================================================

                st.markdown(
                    """
                    <div class="section-header">
                        <div class="section-title">
                            📊 Prediction Analysis
                        </div>
                        <div class="section-description">
                            Interpretation of the model's prediction.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                result_col1, result_col2, result_col3 = st.columns(3)

                with result_col1:

                    st.metric(
                        "Predicted Score",
                        f"{display_prediction:.2f} / 5",
                    )

                with result_col2:

                    st.metric(
                        "Satisfaction Level",
                        category,
                    )

                with result_col3:

                    risk = (
                        "Low"
                        if display_prediction >= 4
                        else "Medium"
                        if display_prediction >= 3
                        else "High"
                    )

                    st.metric(
                        "Negative Review Risk",
                        risk,
                    )


                # ====================================================
                # SCORE PROGRESS
                # ====================================================

                st.markdown("### Satisfaction Score")

                st.progress(
                    display_prediction / 5.0
                )

                if display_prediction >= 4.5:

                    st.success(
                        f"🌟 **{category}** — {message}"
                    )

                elif display_prediction >= 4.0:

                    st.success(
                        f"😊 **{category}** — {message}"
                    )

                elif display_prediction >= 3.0:

                    st.warning(
                        f"😐 **{category}** — {message}"
                    )

                else:

                    st.error(
                        f"⚠️ **{category}** — {message}"
                    )


                # ====================================================
                # INPUT SUMMARY
                # ====================================================

                with st.expander(
                    "📋 View Submitted Input"
                ):

                    st.json(
                        {
                            "payment_sequential": payment_sequential,
                            "payment_installments": payment_installments,
                            "payment_value": payment_value,
                            "price": price,
                            "freight_value": freight_value,
                            "product_name_lenght": product_name_lenght,
                            "product_description_lenght": product_description_lenght,
                            "product_photos_qty": product_photos_qty,
                            "product_weight_g": product_weight_g,
                            "product_length_cm": product_length_cm,
                            "product_height_cm": product_height_cm,
                            "product_width_cm": product_width_cm,
                        }
                    )


                # ====================================================
                # API RESPONSE
                # ====================================================

                with st.expander(
                    "🔍 View MLflow API Response"
                ):

                    st.json(result)


            else:

                st.error(
                    f"❌ Prediction request failed "
                    f"(HTTP {response.status_code})"
                )

                with st.expander(
                    "View server response"
                ):

                    st.code(
                        response.text
                    )


        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to the MLflow prediction server."
            )

            st.info(
                "Start your MLflow deployment first:"
            )

            st.code(
                "python run_deployment.py --config predict",
                language="bash",
            )

            st.caption(
                f"Expected endpoint: {MLFLOW_ENDPOINT}"
            )


        except requests.exceptions.Timeout:

            st.error(
                "⏱️ The MLflow prediction server took too long to respond."
            )

            st.info(
                "Check whether the MLflow deployment service is running."
            )


        except KeyError:

            st.error(
                "❌ Unexpected response received from the MLflow server."
            )

            with st.expander(
                "View raw response"
            ):

                st.code(
                    response.text
                )


        except Exception as e:

            st.error(
                "❌ An unexpected error occurred."
            )

            with st.expander(
                "Technical details"
            ):

                st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="custom-footer">
        Customer Satisfaction MLOps
        &nbsp;•&nbsp;
        ZenML
        &nbsp;•&nbsp;
        MLflow
        &nbsp;•&nbsp;
        LightGBM
        &nbsp;•&nbsp;
        Streamlit
        <br><br>
        Built for production-oriented machine learning experimentation
        and deployment.
    </div>
    """,
    unsafe_allow_html=True,
)