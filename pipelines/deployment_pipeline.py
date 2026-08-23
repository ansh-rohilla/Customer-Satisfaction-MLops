import json
import os

import mlflow
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from steps.clean_data import clean_data
from steps.drift_monitor import drift_monitor
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model

from zenml import pipeline, step
from zenml.constants import (
    DEFAULT_SERVICE_START_STOP_TIMEOUT,
)

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from zenml.integrations.mlflow.services import (
    MLFlowDeploymentService,
)

from zenml.integrations.mlflow.steps import (
    mlflow_model_deployer_step,
)

from .utils import get_data_for_test


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

load_dotenv(
    os.path.join(
        PROJECT_ROOT,
        ".env",
    )
)


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}",
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ============================================================
# MLFLOW MODEL REGISTRY CONFIGURATION
# ============================================================

REGISTERED_MODEL_NAME = (
    "Customer-Satisfaction-Model"
)

MODEL_ALIAS = "champion"


# ============================================================
# MODEL FEATURE ORDER
# ============================================================

MODEL_FEATURES = [
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
# STEP 1: GET DATA FOR INFERENCE
# ============================================================

@step(enable_cache=False)
def dynamic_importer() -> str:

    data = get_data_for_test()

    return data


# ============================================================
# STEP 2: PREPARE INFERENCE DATA
# ============================================================

@step(enable_cache=False)
def prepare_inference_data(
    data: str,
) -> pd.DataFrame:

    data = json.loads(data)

    data.pop("columns", None)
    data.pop("index", None)

    df = pd.DataFrame(
        data["data"],
        columns=MODEL_FEATURES,
    )

    print("\n" + "=" * 60)
    print("INFERENCE DATA PREPARED")
    print("=" * 60)

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print("\nFeatures:")

    for column in df.columns:
        print(f"  - {column}")

    print("=" * 60)

    return df


# ============================================================
# STEP 3: DEPLOYMENT QUALITY GATE
# ============================================================

@step(enable_cache=False)
def deployment_trigger(
    r2_score: float,
    rmse: float,
    min_r2: float,
    max_rmse: float,
) -> bool:

    print("\n" + "=" * 60)
    print("MODEL DEPLOYMENT QUALITY GATE")
    print("=" * 60)

    print(
        f"Test R²              : "
        f"{r2_score:.6f}"
    )

    print(
        f"Minimum required R²  : "
        f"{min_r2:.6f}"
    )

    print(
        f"Test RMSE            : "
        f"{rmse:.6f}"
    )

    print(
        f"Maximum allowed RMSE : "
        f"{max_rmse:.6f}"
    )


    r2_pass = (
        r2_score >= min_r2
    )

    rmse_pass = (
        rmse <= max_rmse
    )


    print("\nQuality Checks:")

    print(
        f"R² Check   : "
        f"{'PASS' if r2_pass else 'FAIL'}"
    )

    print(
        f"RMSE Check : "
        f"{'PASS' if rmse_pass else 'FAIL'}"
    )


    decision = (
        r2_pass
        and rmse_pass
    )


    if decision:

        print(
            "\nDeployment trigger: TRUE"
        )

        print(
            "Model passed all deployment "
            "quality requirements."
        )

    else:

        print(
            "\nDeployment trigger: FALSE"
        )

        print(
            "Model failed one or more "
            "deployment quality requirements."
        )

    print("=" * 60)

    return decision


# ============================================================
# STEP 4: PROMOTE EXACT CANDIDATE TO CHAMPION
# ============================================================

@step(enable_cache=False)
def promote_model_to_champion(
    deployment_decision: bool,
    candidate_version: str,
) -> bool:

    print("\n" + "=" * 60)
    print("MLFLOW MODEL PROMOTION")
    print("=" * 60)

    print(
        f"Registered Model : "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Candidate Version: "
        f"{candidate_version}"
    )

    print(
        f"Target Alias     : "
        f"{MODEL_ALIAS}"
    )


    # ========================================================
    # QUALITY GATE FAILED
    # ========================================================

    if not deployment_decision:

        print(
            "\nQuality gate failed."
        )

        print(
            "Candidate model will NOT "
            "be promoted."
        )

        print(
            "Champion alias remains unchanged."
        )

        print("=" * 60)

        return False


    # ========================================================
    # CREATE MLFLOW CLIENT
    # ========================================================

    client = mlflow.MlflowClient(
        tracking_uri=(
            mlflow.get_tracking_uri()
        )
    )


    # ========================================================
    # VERIFY REGISTERED MODEL
    # ========================================================

    try:

        client.get_registered_model(
            REGISTERED_MODEL_NAME
        )

    except Exception as exc:

        raise RuntimeError(
            f"Registered model "
            f"'{REGISTERED_MODEL_NAME}' "
            f"does not exist."
        ) from exc


    # ========================================================
    # VERIFY EXACT CANDIDATE VERSION
    # ========================================================

    try:

        candidate = (
            client.get_model_version(
                name=REGISTERED_MODEL_NAME,
                version=candidate_version,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            f"Candidate version "
            f"{candidate_version} was not found "
            f"for registered model "
            f"'{REGISTERED_MODEL_NAME}'."
        ) from exc


    print(
        f"\nVerified candidate version: "
        f"{candidate.version}"
    )


    # ========================================================
    # VERIFY CANDIDATE LIFECYCLE
    # ========================================================

    lifecycle = (
        candidate.tags.get(
            "lifecycle"
        )
    )

    if lifecycle != "candidate":

        raise RuntimeError(
            f"Model version "
            f"{candidate_version} is not tagged "
            f"as a candidate."
        )


    # ========================================================
    # PROMOTE EXACT CANDIDATE
    # ========================================================

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
        version=candidate_version,
    )


    # ========================================================
    # UPDATE VERSION TAG
    # ========================================================

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=candidate_version,
        key="lifecycle",
        value="champion",
    )


    # ========================================================
    # VERIFY CHAMPION
    # ========================================================

    champion_version = (
        client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            MODEL_ALIAS,
        )
    )


    if (
        champion_version.version
        != candidate_version
    ):

        raise RuntimeError(
            "Champion verification failed. "
            f"Expected version "
            f"{candidate_version}, "
            f"but alias points to version "
            f"{champion_version.version}."
        )


    print("\nModel successfully promoted.")

    print(
        f"Registered Model : "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Champion Version : "
        f"{champion_version.version}"
    )

    print(
        f"Champion Alias   : "
        f"{MODEL_ALIAS}"
    )

    print("=" * 60)

    return True


# ============================================================
# STEP 5: VERIFY CHAMPION MODEL
# ============================================================

@step(enable_cache=False)
def verify_champion_model(
    deployment_decision: bool,
    candidate_version: str,
) -> bool:

    print("\n" + "=" * 60)
    print("VERIFYING CHAMPION MODEL")
    print("=" * 60)


    # ========================================================
    # SKIP IF QUALITY GATE FAILED
    # ========================================================

    if not deployment_decision:

        print(
            "Deployment was rejected."
        )

        print(
            "Champion verification skipped."
        )

        print("=" * 60)

        return False


    client = mlflow.MlflowClient(
        tracking_uri=(
            mlflow.get_tracking_uri()
        )
    )


    # ========================================================
    # GET CHAMPION
    # ========================================================

    try:

        champion_version = (
            client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                MODEL_ALIAS,
            )
        )

    except Exception as exc:

        print(
            "\nUnable to find champion model."
        )

        print(
            f"Error: {exc}"
        )

        raise RuntimeError(
            f"No model is assigned to alias "
            f"'{MODEL_ALIAS}' for "
            f"'{REGISTERED_MODEL_NAME}'."
        ) from exc


    # ========================================================
    # VERIFY EXACT CANDIDATE WAS PROMOTED
    # ========================================================

    if (
        champion_version.version
        != candidate_version
    ):

        raise RuntimeError(
            "Champion alias points to an "
            "unexpected model version.\n"
            f"Expected: {candidate_version}\n"
            f"Actual: {champion_version.version}"
        )


    print(
        f"Registered Model : "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Champion Alias   : "
        f"{MODEL_ALIAS}"
    )

    print(
        f"Champion Version : "
        f"{champion_version.version}"
    )

    print(
        "\nChampion model verified successfully."
    )

    print("=" * 60)

    return True


# ============================================================
# STEP 6: LOAD DEPLOYED MLFLOW SERVICE
# ============================================================

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:

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
            f"No MLflow prediction service deployed "
            f"by the '{pipeline_step_name}' step "
            f"in the '{pipeline_name}' pipeline "
            f"for the '{model_name}' model "
            f"is currently running."
        )


    print(
        "Found MLflow deployment service:"
    )

    print(existing_services[0])

    return existing_services[0]


# ============================================================
# STEP 7: MAKE PREDICTION
# ============================================================

@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:

    service.start(timeout=10)

    data = json.loads(data)


    # ========================================================
    # REMOVE METADATA
    # ========================================================

    data.pop("columns", None)
    data.pop("index", None)


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        data["data"],
        columns=MODEL_FEATURES,
    )


    # ========================================================
    # CONVERT TO MLFLOW INPUT FORMAT
    # ========================================================

    prediction_data = np.array(
        json.loads(
            df.to_json(
                orient="records"
            )
        )
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = service.predict(
        prediction_data
    )


    print("\nPrediction:")
    print(prediction)

    return prediction


# ============================================================
# PIPELINE 1: CONTINUOUS DEPLOYMENT
# ============================================================

@pipeline(enable_cache=False)
def continuous_deployment_pipeline(
    min_r2: float = 0.08,
    max_rmse: float = 1.50,
    workers: int = 1,
    timeout: int = (
        DEFAULT_SERVICE_START_STOP_TIMEOUT
    ),
):

    """
    Complete continuous deployment pipeline.

    Flow:

        Ingest Data
             ↓
        Clean Data
             ↓
        Model Benchmarking
             ↓
        Optuna Fine-Tuning
             ↓
        Final Model Training
             ↓
        MLflow Candidate Registration
             ↓
        Model Evaluation
             ↓
        Quality Gate
             ↓
        Promote EXACT Candidate
             ↓
        Verify Champion
             ↓
        ZenML MLflow Deployment
    """


    # ========================================================
    # 1. INGEST DATA
    # ========================================================

    df = ingest_data()


    # ========================================================
    # 2. CLEAN AND SPLIT DATA
    # ========================================================

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = clean_data(df)


    # ========================================================
    # 3. TRAIN MODEL
    #
    # IMPORTANT:
    # train_model now returns:
    #
    #     trained_model
    #     candidate_version
    # ========================================================

    (
        model,
        candidate_version,
    ) = train_model(
        x_train,
        x_test,
        y_train,
        y_test,
    )


    # ========================================================
    # 4. EVALUATE MODEL
    # ========================================================

    (
        r2,
        rmse,
        mse,
    ) = evaluation(
        model,
        x_test,
        y_test,
    )


    # ========================================================
    # 5. QUALITY GATE
    # ========================================================

    deployment_decision = (
        deployment_trigger(
            r2_score=r2,
            rmse=rmse,
            min_r2=min_r2,
            max_rmse=max_rmse,
        )
    )


    # ========================================================
    # 6. PROMOTE EXACT CANDIDATE
    # ========================================================

    champion_promoted = (
        promote_model_to_champion(
            deployment_decision=(
                deployment_decision
            ),
            candidate_version=(
                candidate_version
            ),
        )
    )


    # ========================================================
    # 7. VERIFY CHAMPION
    # ========================================================

    champion_verified = (
        verify_champion_model(
            deployment_decision=(
                deployment_decision
            ),
            candidate_version=(
                candidate_version
            ),
        )
    )


    # ========================================================
    # 8. FINAL DEPLOYMENT DECISION
    # ========================================================

    final_deployment_decision = (
        deployment_decision
        and champion_promoted
        and champion_verified
    )


    # ========================================================
    # 9. ZENML + MLFLOW DEPLOYMENT
    # ========================================================

    mlflow_model_deployer_step(
        model=model,
        deploy_decision=(
            final_deployment_decision
        ),
        workers=workers,
        timeout=timeout,
    )


# ============================================================
# PIPELINE 2: INFERENCE + DRIFT MONITORING
# ============================================================

@pipeline(enable_cache=False)
def inference_pipeline(
    pipeline_name: str,
    pipeline_step_name: str,
):

    """
    Monitor incoming data for drift
    and make predictions.

    Flow:

        New Data
           ↓
        Prepare Data
           ↓
        PSI Drift Monitoring
           ↓
        Load Deployed Model
           ↓
        Prediction
    """


    # ========================================================
    # 1. GET NEW INFERENCE DATA
    # ========================================================

    batch_data = dynamic_importer()


    # ========================================================
    # 2. PREPARE INFERENCE DATA
    # ========================================================

    current_data = (
        prepare_inference_data(
            data=batch_data,
        )
    )


    # ========================================================
    # 3. DATA DRIFT MONITORING
    # ========================================================

    drift_detected = drift_monitor(
        data=current_data,
        model_name=REGISTERED_MODEL_NAME,
        model_alias=MODEL_ALIAS,
    )


    # ========================================================
    # 4. LOAD DEPLOYED MODEL
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
    # 5. MAKE PREDICTION
    # ========================================================

    predictor(
        service=model_deployment_service,
        data=batch_data,
    )