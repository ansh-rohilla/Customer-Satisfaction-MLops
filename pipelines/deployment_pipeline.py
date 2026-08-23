import json

import mlflow
import numpy as np
import pandas as pd

from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model

from zenml import pipeline, step
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from .utils import get_data_for_test


# ============================================================
# STEP 1: Get data for inference
# ============================================================

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Get the latest data for inference."""

    data = get_data_for_test()

    return data


# ============================================================
# STEP 2: Deployment Quality Gate
# ============================================================

@step
def deployment_trigger(
    r2_score: float,
    rmse: float,
    min_r2: float,
    max_rmse: float,
) -> bool:
    """
    Decide whether the model should be deployed.

    Deployment happens only when BOTH conditions are met:

        R² >= minimum required R²
        RMSE <= maximum allowed RMSE
    """

    print("\n" + "=" * 60)
    print("MODEL DEPLOYMENT QUALITY GATE")
    print("=" * 60)

    print(f"Test R²: {r2_score:.6f}")
    print(f"Minimum required R²: {min_r2:.6f}")

    print(f"Test RMSE: {rmse:.6f}")
    print(f"Maximum allowed RMSE: {max_rmse:.6f}")

    # ========================================================
    # QUALITY CHECKS
    # ========================================================

    r2_pass = r2_score >= min_r2
    rmse_pass = rmse <= max_rmse

    print("\nQuality Checks:")

    print(
        f"R² Check   : "
        f"{'PASS' if r2_pass else 'FAIL'}"
    )

    print(
        f"RMSE Check : "
        f"{'PASS' if rmse_pass else 'FAIL'}"
    )

    # ========================================================
    # FINAL DEPLOYMENT DECISION
    # ========================================================

    decision = r2_pass and rmse_pass

    if decision:
        print("\nDeployment trigger: TRUE")
        print(
            "Model passed all deployment "
            "quality requirements."
        )
    else:
        print("\nDeployment trigger: FALSE")
        print(
            "Model failed one or more "
            "deployment quality requirements."
        )

    print("=" * 60)

    return decision

    # ========================================================
    # LOG QUALITY GATE TO MLFLOW
    # ========================================================

    mlflow.log_metric(
        "deployment_test_r2",
        float(r2_score),
    )

    mlflow.log_metric(
        "deployment_test_rmse",
        float(rmse),
    )

    mlflow.log_param(
        "deployment_min_r2",
        min_r2,
    )

    mlflow.log_param(
        "deployment_max_rmse",
        max_rmse,
    )

    mlflow.log_param(
        "deployment_r2_pass",
        r2_pass,
    )

    mlflow.log_param(
        "deployment_rmse_pass",
        rmse_pass,
    )

    mlflow.log_param(
        "deployment_decision",
        decision,
    )

    print("=" * 60)

    return decision


# ============================================================
# STEP 3: Load deployed MLflow service
# ============================================================

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """
    Get the MLflow prediction service deployed by ZenML.
    """

    model_deployer = (
        MLFlowModelDeployer
        .get_active_model_deployer()
    )

    existing_services = (
        model_deployer.find_model_server(
            pipeline_name=pipeline_name,
            pipeline_step_name=pipeline_step_name,
            model_name=model_name,
            running=running,
        )
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by "
            f"the '{pipeline_step_name}' step in the "
            f"'{pipeline_name}' pipeline for the "
            f"'{model_name}' model is currently running."
        )

    print("Found MLflow deployment service:")
    print(existing_services[0])

    return existing_services[0]


# ============================================================
# STEP 4: Make prediction
# ============================================================

@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run inference against the deployed MLflow model."""

    service.start(timeout=10)

    data = json.loads(data)

    # Remove metadata
    data.pop("columns", None)
    data.pop("index", None)

    columns_for_df = [
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

    df = pd.DataFrame(
        data["data"],
        columns=columns_for_df,
    )

    prediction_data = np.array(
        json.loads(
            df.to_json(
                orient="records"
            )
        )
    )

    prediction = service.predict(
        prediction_data
    )

    print("Prediction:")
    print(prediction)

    return prediction


# ============================================================
# PIPELINE 1: Continuous Deployment
# ============================================================

@pipeline(enable_cache=False)
def continuous_deployment_pipeline(
    min_r2: float = 0.08,
    max_rmse: float = 1.50,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    """
    Train, benchmark, tune, evaluate and
    conditionally deploy the best model.

    Pipeline flow:

        Ingest
          ↓
        Clean
          ↓
        Benchmark Models
          ↓
        Select Best Model
          ↓
        Optuna Fine-Tuning
          ↓
        Evaluation
          ↓
        Deployment Quality Gate
          ↓
        MLflow Deployment
    """

    # ========================================================
    # 1. Ingest data
    # ========================================================

    df = ingest_data()

    # ========================================================
    # 2. Clean and split data
    # ========================================================

    x_train, x_test, y_train, y_test = clean_data(
        df
    )

    # ========================================================
    # 3. Automatically benchmark and train best model
    # ========================================================

    model = train_model(
        x_train,
        x_test,
        y_train,
        y_test,
    )

    # ========================================================
    # 4. Evaluate best model
    # ========================================================

    r2, rmse, mse = evaluation(
        model,
        x_test,
        y_test,
    )

    # ========================================================
    # 5. Deployment Quality Gate
    # ========================================================

    deployment_decision = deployment_trigger(
        r2_score=r2,
        rmse=rmse,
        min_r2=min_r2,
        max_rmse=max_rmse,
    )

    # ========================================================
    # 6. Deploy best model using MLflow
    # ========================================================

    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )


# ============================================================
# PIPELINE 2: Inference
# ============================================================

@pipeline(enable_cache=False)
def inference_pipeline(
    pipeline_name: str,
    pipeline_step_name: str,
):
    """
    Load the deployed MLflow model
    and make predictions.
    """

    # ========================================================
    # 1. Get new inference data
    # ========================================================

    batch_data = dynamic_importer()

    # ========================================================
    # 2. Get deployed MLflow service
    # ========================================================

    model_deployment_service = (
        prediction_service_loader(
            pipeline_name=pipeline_name,
            pipeline_step_name=pipeline_step_name,
            running=True,
            model_name="model",
        )
    )

    # ========================================================
    # 3. Make prediction
    # ========================================================

    predictor(
        service=model_deployment_service,
        data=batch_data,
    )