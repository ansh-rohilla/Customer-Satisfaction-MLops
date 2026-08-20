import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MLFLOW_ENDPOINT = "http://127.0.0.1:8000/invocations"

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


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Satisfaction Predictor",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .prediction-card {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #667eea 0%,
            #764ba2 100%
        );
        color: white;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-value {
        font-size: 52px;
        font-weight: 700;
    }

    .prediction-label {
        font-size: 18px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .status-card {
        padding: 15px;
        border-radius: 12px;
        background-color: #f5f5f5;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="title">⭐ Customer Satisfaction Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Predict customer review satisfaction using the deployed
    machine learning model powered by ZenML + MLflow.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Model:** LightGBM Regressor")

    st.write("**Framework:** ZenML + MLflow")

    st.write("**Prediction Type:** Regression")

    st.write("**Output:** Customer Review Score")

    st.divider()

    st.subheader("🚀 Model Server")

    st.code(
        "http://127.0.0.1:8000/invocations",
        language="text",
    )


# --------------------------------------------------
# Input section
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📦 Order Information</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:

    payment_sequential = st.number_input(
        "Payment Sequential",
        min_value=0,
        value=1,
        step=1,
    )

    payment_installments = st.number_input(
        "Payment Installments",
        min_value=0,
        value=1,
        step=1,
    )

    payment_value = st.number_input(
        "Payment Value",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    price = st.number_input(
        "Product Price",
        min_value=0.0,
        value=80.0,
        step=10.0,
    )


with col2:

    freight_value = st.number_input(
        "Freight Value",
        min_value=0.0,
        value=20.0,
        step=5.0,
    )

    product_name_lenght = st.number_input(
        "Product Name Length",
        min_value=0,
        value=40,
        step=1,
    )

    product_description_lenght = st.number_input(
        "Product Description Length",
        min_value=0,
        value=200,
        step=10,
    )

    product_photos_qty = st.number_input(
        "Product Photos Quantity",
        min_value=0,
        value=3,
        step=1,
    )


with col3:

    product_weight_g = st.number_input(
        "Product Weight (g)",
        min_value=0.0,
        value=500.0,
        step=50.0,
    )

    product_length_cm = st.number_input(
        "Product Length (cm)",
        min_value=0.0,
        value=20.0,
        step=1.0,
    )

    product_height_cm = st.number_input(
        "Product Height (cm)",
        min_value=0.0,
        value=10.0,
        step=1.0,
    )

    product_width_cm = st.number_input(
        "Product Width (cm)",
        min_value=0.0,
        value=15.0,
        step=1.0,
    )


# --------------------------------------------------
# Prediction button
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Satisfaction",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

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

    with st.spinner("Running prediction..."):

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

                # Keep prediction within normal review range
                display_prediction = max(
                    1.0,
                    min(5.0, prediction)
                )

                # ------------------------------------------
                # Prediction card
                # ------------------------------------------

                st.markdown(
                    f"""
                    <div class="prediction-card">

                        <div class="prediction-label">
                            Predicted Customer Satisfaction
                        </div>

                        <div class="prediction-value">
                            ⭐ {display_prediction:.2f} / 5
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ------------------------------------------
                # Interpretation
                # ------------------------------------------

                st.subheader("📊 Result Interpretation")

                if display_prediction >= 4.5:

                    st.success(
                        "Excellent satisfaction — the customer "
                        "is likely to give a very positive review."
                    )

                elif display_prediction >= 4.0:

                    st.success(
                        "Very good satisfaction — the customer "
                        "is likely to be satisfied."
                    )

                elif display_prediction >= 3.0:

                    st.warning(
                        "Moderate satisfaction — there is room "
                        "for improvement."
                    )

                else:

                    st.error(
                        "Low satisfaction — the order may result "
                        "in a negative customer review."
                    )

                # ------------------------------------------
                # Model response
                # ------------------------------------------

                with st.expander("🔍 View API Response"):

                    st.json(result)

            else:

                st.error(
                    f"Prediction failed. "
                    f"HTTP Status: {response.status_code}"
                )

                st.code(response.text)

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to the MLflow model server."
            )

            st.info(
                "Make sure the deployment is running first:\n\n"
                "python run_deployment.py --config deploy"
            )

        except Exception as e:

            st.error(
                f"An unexpected error occurred: {str(e)}"
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Satisfaction MLOps • ZenML • MLflow • LightGBM • Streamlit"
)