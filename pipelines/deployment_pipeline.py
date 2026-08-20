import json

import numpy as np
import pandas as pd

from materializer.custom_materializer import cs_materializer
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


# Step 1: Get data for inference


@step(enable_cache=True)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""

    data = get_data_for_test()
    return data

# Step 2: Deployment trigger


class DeploymentTriggerConfig:
    """Configuration used to decide whether a model should be deployed."""

    def __init__(self, min_r2: float = 0.5):
        self.min_r2 = min_r2


@step
def deployment_trigger(
    r2_score: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """
    Deploy the model only if its R² score reaches the required threshold.

    R² is appropriate here because this is a regression problem:
    higher R² = better model.
    """

    print(f"Model R² score: {r2_score}")
    print(f"Minimum required R²: {config.min_r2}")

    decision = r2_score >= config.min_r2

    if decision:
        print("Deployment trigger: TRUE")
        print("Model meets the required R² threshold.")
    else:
        print("Deployment trigger: FALSE")
        print("Model does not meet the required R² threshold.")

    return decision



# Step 3: Load deployed MLflow service


@step(enable_cache=True)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """
    Get the MLflow prediction service started by the deployment pipeline.
    """

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"'{pipeline_step_name}' step in the "
            f"'{pipeline_name}' pipeline for the "
            f"'{model_name}' model is currently running."
        )

    print(existing_services)
    print(type(existing_services))

    return existing_services[0]


# Step 4: Make prediction


@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against the deployed MLflow model."""

    service.start(timeout=10)

    data = json.loads(data)

    # Remove metadata fields
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

    json_list = json.loads(
        json.dumps(
            list(df.T.to_dict().values())
        )
    )

    prediction_data = np.array(json_list)

    prediction = service.predict(prediction_data)

    print("Prediction:")
    print(prediction)

    return prediction



# Pipeline 1: Continuous deployment


@pipeline(enable_cache=True)
def continuous_deployment_pipeline(
    min_r2: float = 0.5,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    """
    Train, evaluate and conditionally deploy the model.

    Deployment happens only when R² >= min_r2.
    """

    # 1. Ingest data
    df = ingest_data()

    # 2. Clean and split data
    x_train, x_test, y_train, y_test = clean_data(df)

    # 3. Train model
    model = train_model(
        x_train,
        x_test,
        y_train,
        y_test,
    )

    # 4. Evaluate model
    mse, r2, rmse = evaluation(
        model,
        x_test,
        y_test,
    )

    # 5. Decide whether to deploy
    deployment_decision = deployment_trigger(
        r2_score=r2,
        config=DeploymentTriggerConfig(
            min_r2=min_r2
        ),
    )

    # 6. Deploy using MLflow
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )



# Pipeline 2: Inference


@pipeline(enable_cache=True)
def inference_pipeline(
    pipeline_name: str,
    pipeline_step_name: str,
):
    """
    Load the deployed MLflow model and make predictions.
    """

    # Get new inference data
    batch_data = dynamic_importer()

    # Get deployed MLflow service
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
        model_name="model",
    )

    # Make prediction
    predictor(
        service=model_deployment_service,
        data=batch_data,
    )