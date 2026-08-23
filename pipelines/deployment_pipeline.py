import json
import os
import time
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from mlflow import MlflowClient

from zenml import pipeline, step
from zenml.client import Client
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.clean_data import clean_data
from steps.drift_monitor import drift_monitor
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model

from .utils import get_data_for_test


# ============================================================
# PROJECT / ENVIRONMENT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(
    os.path.join(PROJECT_ROOT, ".env")
)


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}",
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ============================================================
# ZENML MLFLOW EXPERIMENT TRACKER
# ============================================================

experiment_tracker = (
    Client()
    .active_stack
    .experiment_tracker
)

if experiment_tracker is None:
    raise RuntimeError(
        "No active ZenML experiment tracker was found."
    )


# ============================================================
# MODEL REGISTRY CONFIGURATION
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
# DEPLOYMENT CONFIGURATION
# ============================================================

DEPLOYMENT_PIPELINE_NAME = (
    "continuous_deployment_pipeline"
)

DEPLOYMENT_STEP_NAME = (
    "mlflow_model_deployer_step"
)

DEPLOYMENT_MODEL_NAME = "model"


# ============================================================
# HELPER: PRINT MLFLOW CONFIGURATION
# ============================================================

def _print_mlflow_configuration(
    context: str,
) -> None:

    print("\n" + "=" * 60)
    print(
        f"MLFLOW CONFIGURATION - {context}"
    )
    print("=" * 60)

    print(
        f"Tracking URI     : "
        f"{mlflow.get_tracking_uri()}"
    )

    print(
        f"Registered Model : "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Model Alias      : "
        f"{MODEL_ALIAS}"
    )

    print("=" * 60)


# ============================================================
# HELPER: GET PREDICTION URL
# ============================================================

def _get_prediction_url(
    service: MLFlowDeploymentService,
) -> str:
    """
    Resolve the prediction URL from a ZenML MLflow
    deployment service.

    Supports different ZenML versions.
    """

    get_url_method = getattr(
        service,
        "get_prediction_url",
        None,
    )

    if callable(get_url_method):

        try:
            url = get_url_method()

            if url:
                return str(url)

        except Exception:
            pass

    prediction_url = getattr(
        service,
        "prediction_url",
        None,
    )

    if prediction_url:
        return str(prediction_url)

    endpoint = getattr(
        service,
        "endpoint",
        None,
    )

    if endpoint is not None:

        endpoint_url = getattr(
            endpoint,
            "prediction_url",
            None,
        )

        if endpoint_url:
            return str(endpoint_url)

    status = getattr(
        service,
        "status",
        None,
    )

    if status is not None:

        status_url = getattr(
            status,
            "prediction_url",
            None,
        )

        if status_url:
            return str(status_url)

    raise RuntimeError(
        "Could not determine the MLflow prediction URL."
    )


# ============================================================
# HELPER: GET SERVICE ERROR
# ============================================================

def _get_service_error(
    service: MLFlowDeploymentService,
) -> Optional[str]:
    """
    Extract the most useful error message from
    a ZenML deployment service.
    """

    status = getattr(
        service,
        "status",
        None,
    )

    if status is None:
        return None

    last_error = getattr(
        status,
        "last_error",
        None,
    )

    if last_error:
        return str(last_error)

    error = getattr(
        status,
        "error",
        None,
    )

    if error:
        return str(error)

    return None


# ============================================================
# STEP 1: GET INFERENCE DATA
# ============================================================

@step(enable_cache=False)
def dynamic_importer() -> str:

    data = get_data_for_test()

    if not isinstance(data, str):

        raise TypeError(
            "Inference data must be returned "
            "as a JSON string."
        )

    return data


# ============================================================
# STEP 2: PREPARE INFERENCE DATA
# ============================================================

@step(enable_cache=False)
def prepare_inference_data(
    data: str,
) -> pd.DataFrame:

    try:

        payload = json.loads(data)

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Inference payload is not valid JSON."
        ) from exc

    if not isinstance(payload, dict):

        raise TypeError(
            "Inference payload must be a JSON object."
        )

    if "data" not in payload:

        raise ValueError(
            "Inference payload does not contain "
            "a 'data' field."
        )

    raw_data = payload["data"]

    if not isinstance(raw_data, list):

        raise TypeError(
            "Inference 'data' must be a list of rows."
        )

    if not raw_data:

        raise ValueError(
            "Inference dataset is empty."
        )

    expected_feature_count = len(
        MODEL_FEATURES
    )

    for row_index, row in enumerate(raw_data):

        if not isinstance(
            row,
            (list, tuple),
        ):

            raise TypeError(
                f"Inference row {row_index} "
                f"must be a list or tuple."
            )

        if len(row) != expected_feature_count:

            raise ValueError(
                f"Inference row {row_index} contains "
                f"{len(row)} features. "
                f"Expected {expected_feature_count}."
            )

    df = pd.DataFrame(
        raw_data,
        columns=MODEL_FEATURES,
    )

    if list(df.columns) != MODEL_FEATURES:

        raise ValueError(
            "Inference feature order does not match "
            "the trained model."
        )

    for column in MODEL_FEATURES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if (
        df[MODEL_FEATURES]
        .isnull()
        .any()
        .any()
    ):

        invalid_columns = (
            df[MODEL_FEATURES]
            .columns[
                df[MODEL_FEATURES]
                .isnull()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "Inference data contains missing or "
            "non-numeric values in columns: "
            f"{invalid_columns}"
        )

    values = (
        df[MODEL_FEATURES]
        .to_numpy(dtype=np.float64)
    )

    if not np.isfinite(values).all():

        raise ValueError(
            "Inference data contains infinite values."
        )

    print("\n" + "=" * 60)
    print("INFERENCE DATA PREPARED")
    print("=" * 60)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    print("\nFeatures:")

    for column in MODEL_FEATURES:
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
        f"Test R²              : {r2_score:.6f}"
    )

    print(
        f"Minimum required R²  : {min_r2:.6f}"
    )

    print(
        f"Test RMSE            : {rmse:.6f}"
    )

    print(
        f"Maximum allowed RMSE : {max_rmse:.6f}"
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

    print(
        f"\nDeployment trigger: "
        f"{decision}"
    )

    if decision:

        print(
            "Model passed all deployment "
            "quality requirements."
        )

    else:

        print(
            "Model failed one or more deployment "
            "quality requirements."
        )

    print("=" * 60)

    return decision


# ============================================================
# STEP 4: PROMOTE EXACT CANDIDATE TO CHAMPION
# ============================================================

@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
)
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

    if not deployment_decision:

        print("\nQuality gate failed.")

        print(
            "Candidate will NOT be promoted."
        )

        print(
            "Champion remains unchanged."
        )

        print("=" * 60)

        return False

    _print_mlflow_configuration(
        "PROMOTION STEP"
    )

    tracking_uri = (
        mlflow.get_tracking_uri()
    )

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    # --------------------------------------------------------
    # VERIFY REGISTERED MODEL
    # --------------------------------------------------------

    try:

        client.get_registered_model(
            REGISTERED_MODEL_NAME
        )

    except Exception as exc:

        raise RuntimeError(
            f"Registered model "
            f"'{REGISTERED_MODEL_NAME}' does not exist.\n"
            f"Tracking URI: {tracking_uri}"
        ) from exc

    # --------------------------------------------------------
    # VERIFY EXACT CANDIDATE
    # --------------------------------------------------------

    try:

        candidate = (
            client.get_model_version(
                name=REGISTERED_MODEL_NAME,
                version=str(candidate_version),
            )
        )

    except Exception as exc:

        raise RuntimeError(
            f"Candidate version "
            f"{candidate_version} was not found "
            f"for '{REGISTERED_MODEL_NAME}'."
        ) from exc

    print(
        f"\nVerified candidate version: "
        f"{candidate.version}"
    )

    print(
        f"Candidate Run ID: "
        f"{candidate.run_id}"
    )

    # --------------------------------------------------------
    # VERIFY CANDIDATE LIFECYCLE
    # --------------------------------------------------------

    lifecycle = candidate.tags.get(
        "lifecycle"
    )

    print(
        f"Candidate Lifecycle: "
        f"{lifecycle}"
    )

    if (
        lifecycle is not None
        and lifecycle != "candidate"
    ):

        raise RuntimeError(
            f"Model version {candidate_version} "
            f"is not tagged as a candidate.\n"
            f"Current lifecycle: {lifecycle}"
        )

    # --------------------------------------------------------
    # PROMOTE EXACT VERSION
    # --------------------------------------------------------

    print(
        f"\nPromoting version "
        f"{candidate_version} "
        f"to alias '{MODEL_ALIAS}'..."
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
        version=str(candidate_version),
    )

    # --------------------------------------------------------
    # UPDATE LIFECYCLE
    # --------------------------------------------------------

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=str(candidate_version),
        key="lifecycle",
        value="champion",
    )

    # --------------------------------------------------------
    # VERIFY ALIAS
    # --------------------------------------------------------

    champion_version = (
        client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            MODEL_ALIAS,
        )
    )

    if (
        str(champion_version.version)
        != str(candidate_version)
    ):

        raise RuntimeError(
            "Champion verification failed.\n"
            f"Expected version: {candidate_version}\n"
            f"Actual version: "
            f"{champion_version.version}"
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

    print(
        f"Tracking URI     : "
        f"{tracking_uri}"
    )

    print("=" * 60)

    return True


# ============================================================
# STEP 5: VERIFY CHAMPION
# ============================================================

@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
)
def verify_champion_model(
    deployment_decision: bool,
    candidate_version: str,
) -> bool:

    print("\n" + "=" * 60)
    print("VERIFYING CHAMPION MODEL")
    print("=" * 60)

    if not deployment_decision:

        print(
            "Deployment was rejected."
        )

        print(
            "Champion verification skipped."
        )

        print("=" * 60)

        return False

    _print_mlflow_configuration(
        "CHAMPION VERIFICATION"
    )

    tracking_uri = (
        mlflow.get_tracking_uri()
    )

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    try:

        champion_version = (
            client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                MODEL_ALIAS,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            f"No model is assigned to alias "
            f"'{MODEL_ALIAS}' for "
            f"'{REGISTERED_MODEL_NAME}'."
        ) from exc

    if (
        str(champion_version.version)
        != str(candidate_version)
    ):

        raise RuntimeError(
            "Champion alias points to an "
            "unexpected model version.\n"
            f"Expected: {candidate_version}\n"
            f"Actual: {champion_version.version}"
        )

    lifecycle = (
        champion_version.tags.get(
            "lifecycle"
        )
    )

    if lifecycle != "champion":

        raise RuntimeError(
            "Champion model has an invalid "
            "lifecycle tag.\n"
            f"Expected: champion\n"
            f"Actual: {lifecycle}"
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
        f"Lifecycle        : "
        f"{lifecycle}"
    )

    print(
        f"Tracking URI     : "
        f"{tracking_uri}"
    )

    print(
        "\nChampion model verified successfully."
    )

    print("=" * 60)

    return True


# ============================================================
# STEP 6: COMBINE DEPLOYMENT DECISIONS
# ============================================================

@step(enable_cache=False)
def combine_deployment_decisions(
    deployment_decision: bool,
    champion_promoted: bool,
    champion_verified: bool,
) -> bool:

    print("\n" + "=" * 60)
    print("FINAL DEPLOYMENT DECISION")
    print("=" * 60)

    final_decision = (
        deployment_decision
        and champion_promoted
        and champion_verified
    )

    print(
        f"Quality Gate       : "
        f"{deployment_decision}"
    )

    print(
        f"Champion Promoted  : "
        f"{champion_promoted}"
    )

    print(
        f"Champion Verified  : "
        f"{champion_verified}"
    )

    print(
        f"Final Deployment   : "
        f"{final_decision}"
    )

    print("=" * 60)

    return final_decision


# ============================================================
# STEP 7: LOAD DEPLOYMENT SERVICE
# ============================================================

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    model_name: str = DEPLOYMENT_MODEL_NAME,
) -> MLFlowDeploymentService:

    print("\n" + "=" * 60)
    print("LOADING MLFLOW DEPLOYMENT SERVICE")
    print("=" * 60)

    print(f"Pipeline        : {pipeline_name}")
    print(f"Deployment Step : {pipeline_step_name}")
    print(f"Model Name      : {model_name}")

    # --------------------------------------------------------
    # GET MLFLOW MODEL DEPLOYER
    # --------------------------------------------------------

    model_deployer = (
        MLFlowModelDeployer
        .get_active_model_deployer()
    )

    # --------------------------------------------------------
    # FIND EXISTING SERVICE
    # --------------------------------------------------------

    existing_services = (
        model_deployer.find_model_server(
            pipeline_name=pipeline_name,
            pipeline_step_name=pipeline_step_name,
            model_name=model_name,
        )
    )

    if not existing_services:

        raise RuntimeError(
            "\nNo MLflow deployment service was found.\n\n"
            f"Pipeline : {pipeline_name}\n"
            f"Step     : {pipeline_step_name}\n"
            f"Model    : {model_name}\n\n"
            "Run the continuous deployment pipeline first."
        )

    # --------------------------------------------------------
    # SELECT SERVICE
    # --------------------------------------------------------

    service = existing_services[0]

    print("\nFound MLflow deployment service:")
    print(service)

    # --------------------------------------------------------
    # CHECK CURRENT STATE
    # --------------------------------------------------------

    try:

        if service.is_running:

            print("\nService is already running.")

        else:

            print("\nService is NOT running.")

            # ------------------------------------------------
            # CHECK FAILURE STATE BEFORE STARTING
            # ------------------------------------------------

            if service.is_failed:

                print("\nMLflow service is in FAILED state.")

                try:
                    print(
                        "Service status:"
                    )
                    print(
                        service.status
                    )
                except Exception:
                    pass

                raise RuntimeError(
                    "The existing MLflow deployment service "
                    "is in a failed state. "
                    "The service must be recreated."
                )

            # ------------------------------------------------
            # START SERVICE
            # ------------------------------------------------

            print(
                "\nStarting MLflow deployment service..."
            )

            service.start(
                timeout=60
            )

            print(
                "Start command completed."
            )

            # ------------------------------------------------
            # VERIFY AFTER START
            # ------------------------------------------------

            if not service.is_running:

                print(
                    "\nService did not become running."
                )

                # Print detailed status information
                try:

                    print(
                        "\nService status:"
                    )

                    print(
                        service.status
                    )

                except Exception:
                    pass

                if service.is_failed:

                    raise RuntimeError(
                        "\nMLflow deployment service "
                        "entered FAILED state after "
                        "attempting to start."
                    )

                raise RuntimeError(
                    "\nMLflow deployment service "
                    "did not become running after "
                    "the start command."
                )

            print(
                "\nMLflow deployment service "
                "started successfully."
            )

    except RuntimeError:
        raise

    except Exception as exc:

        raise RuntimeError(
            "Unable to start or verify the MLflow "
            "deployment service.\n"
            f"Error: {exc}"
        ) from exc

    # --------------------------------------------------------
    # GET PREDICTION URL
    # --------------------------------------------------------

    try:

        prediction_url = (
            _get_prediction_url(
                service
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "MLflow deployment service is running, "
            "but its prediction URL could not be determined."
        ) from exc

    print(
        f"\nPrediction URL:"
    )

    print(
        prediction_url
    )

    # --------------------------------------------------------
    # FINAL VERIFICATION
    # --------------------------------------------------------

    try:

        if not service.is_running:

            raise RuntimeError(
                "MLflow deployment service stopped "
                "before inference could begin."
            )

    except AttributeError:
        pass

    print(
        "\nMLflow deployment service is ready."
    )

    print("=" * 60)

    return service


# ============================================================
# STEP 8: MAKE PREDICTION
# ============================================================

@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: pd.DataFrame,
) -> np.ndarray:

    print("\n" + "=" * 60)
    print("STARTING PREDICTION")
    print("=" * 60)

    if not isinstance(
        data,
        pd.DataFrame,
    ):

        raise TypeError(
            "Prediction input must be "
            "a pandas DataFrame."
        )

    if data.empty:

        raise ValueError(
            "Prediction dataset is empty."
        )

    if list(data.columns) != MODEL_FEATURES:

        raise ValueError(
            "Prediction feature order does not "
            "match the trained model.\n"
            f"Expected: {MODEL_FEATURES}\n"
            f"Received: {list(data.columns)}"
        )

    prediction_data = data[
        MODEL_FEATURES
    ].copy()

    for column in MODEL_FEATURES:

        prediction_data[column] = (
            pd.to_numeric(
                prediction_data[column],
                errors="coerce",
            )
        )

    if (
        prediction_data
        .isnull()
        .any()
        .any()
    ):

        invalid_columns = (
            prediction_data.columns[
                prediction_data
                .isnull()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "Prediction data contains missing "
            "or non-numeric values in columns: "
            f"{invalid_columns}"
        )

    numeric_values = (
        prediction_data
        .to_numpy(dtype=np.float64)
    )

    if not np.isfinite(
        numeric_values
    ).all():

        raise ValueError(
            "Prediction data contains "
            "infinite values."
        )

    # --------------------------------------------------------
    # GET PREDICTION URL
    # --------------------------------------------------------

    try:

        prediction_url = (
            _get_prediction_url(
                service
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not obtain MLflow "
            "prediction URL."
        ) from exc

    print(
        "\nMLflow prediction URL:"
    )

    print(
        prediction_url
    )

    # --------------------------------------------------------
    # BUILD DATAFRAME_SPLIT PAYLOAD
    # --------------------------------------------------------

    payload = {
        "dataframe_split": {
            "columns": MODEL_FEATURES,
            "data": numeric_values.tolist(),
        }
    }

    print(
        "\nSending dataframe_split payload "
        "to MLflow model..."
    )

    print(
        f"Payload rows    : "
        f"{len(prediction_data)}"
    )

    print(
        f"Payload columns : "
        f"{len(MODEL_FEATURES)}"
    )

    # --------------------------------------------------------
    # HTTP REQUEST
    # --------------------------------------------------------

    try:

        response = requests.post(
            url=prediction_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    except requests.exceptions.RequestException as exc:

        raise RuntimeError(
            "MLflow model prediction HTTP "
            "request failed.\n"
            f"URL: {prediction_url}\n"
            f"Error: {exc}"
        ) from exc

    if not response.ok:

        print("\n" + "=" * 60)
        print("MLFLOW SERVER ERROR")
        print("=" * 60)

        print(
            f"HTTP Status : "
            f"{response.status_code}"
        )

        print(
            f"URL         : "
            f"{prediction_url}"
        )

        print("\nResponse:")

        print(
            response.text
        )

        print("=" * 60)

        raise RuntimeError(
            "MLflow prediction server returned "
            f"HTTP {response.status_code}."
        )

    # --------------------------------------------------------
    # PARSE RESPONSE
    # --------------------------------------------------------

    try:

        response_json = response.json()

    except ValueError as exc:

        raise RuntimeError(
            "MLflow returned a non-JSON response.\n"
            f"Response: {response.text}"
        ) from exc

    print(
        "\nMLflow response received:"
    )

    print(
        response_json
    )

    # --------------------------------------------------------
    # EXTRACT PREDICTIONS
    # --------------------------------------------------------

    if isinstance(
        response_json,
        dict,
    ):

        if "predictions" in response_json:

            prediction = np.asarray(
                response_json["predictions"]
            )

        elif "outputs" in response_json:

            prediction = np.asarray(
                response_json["outputs"]
            )

        else:

            raise RuntimeError(
                "MLflow response does not contain "
                "'predictions' or 'outputs'.\n"
                f"Response: {response_json}"
            )

    elif isinstance(
        response_json,
        list,
    ):

        prediction = np.asarray(
            response_json
        )

    else:

        raise RuntimeError(
            "Unexpected MLflow response format.\n"
            f"Response type: "
            f"{type(response_json).__name__}\n"
            f"Response: {response_json}"
        )

    # --------------------------------------------------------
    # FLATTEN PREDICTIONS
    # --------------------------------------------------------

    prediction = np.asarray(
        prediction
    ).reshape(-1)

    # --------------------------------------------------------
    # VALIDATE OUTPUT COUNT
    # --------------------------------------------------------

    if len(prediction) != len(data):

        raise RuntimeError(
            "Number of predictions does not "
            "match number of input rows.\n"
            f"Input rows: {len(data)}\n"
            f"Predictions: {len(prediction)}"
        )

    # --------------------------------------------------------
    # VALIDATE OUTPUT VALUES
    # --------------------------------------------------------

    try:

        prediction_float = (
            prediction.astype(
                np.float64
            )
        )

    except (TypeError, ValueError) as exc:

        raise RuntimeError(
            "MLflow returned non-numeric "
            "prediction values."
        ) from exc

    if not np.isfinite(
        prediction_float
    ).all():

        raise RuntimeError(
            "MLflow returned invalid "
            "prediction values."
        )

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    print(
        f"Number of predictions: "
        f"{len(prediction_float)}"
    )

    print(
        f"Prediction shape     : "
        f"{prediction_float.shape}"
    )

    print("\nPredictions:")

    print(
        prediction_float
    )

    print("=" * 60)

    return prediction_float


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
        Train Model
             ↓
        Evaluate Model
             ↓
        Quality Gate
             ↓
        Promote Candidate
             ↓
        Verify Champion
             ↓
        Final Deployment Decision
             ↓
        MLflow Deployment
    """

    # ========================================================
    # 1. INGEST
    # ========================================================

    df = ingest_data()

    # ========================================================
    # 2. CLEAN
    # ========================================================

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = clean_data(df)

    # ========================================================
    # 3. TRAIN
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
    # 4. EVALUATE
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
    # 8. FINAL DECISION
    # ========================================================

    final_deployment_decision = (
        combine_deployment_decisions(
            deployment_decision=(
                deployment_decision
            ),
            champion_promoted=(
                champion_promoted
            ),
            champion_verified=(
                champion_verified
            ),
        )
    )

    # ========================================================
    # 9. DEPLOY
    # ========================================================

    mlflow_model_deployer_step(
        model=model,
        deploy_decision=(
            final_deployment_decision
        ),
        model_name=DEPLOYMENT_MODEL_NAME,
        workers=workers,
        timeout=timeout,
    )


# ============================================================
# PIPELINE 2: INFERENCE + DRIFT MONITORING
# ============================================================

@pipeline(enable_cache=False)
def inference_pipeline(
    pipeline_name: str = (
        DEPLOYMENT_PIPELINE_NAME
    ),
    pipeline_step_name: str = (
        DEPLOYMENT_STEP_NAME
    ),
):
    """
    Inference and drift monitoring pipeline.

    Flow:

        New Data
           ↓
        Prepare Data
           ↓
        PSI Drift Monitoring
           ↓
        Load Champion Deployment
           ↓
        Prediction
    """

    # ========================================================
    # 1. GET NEW DATA
    # ========================================================

    batch_data = dynamic_importer()

    # ========================================================
    # 2. PREPARE DATA
    # ========================================================

    current_data = (
        prepare_inference_data(
            data=batch_data,
        )
    )

    # ========================================================
    # 3. DRIFT MONITORING
    # ========================================================

    drift_detected = drift_monitor(
        data=current_data,
        model_name=(
            REGISTERED_MODEL_NAME
        ),
        model_alias=MODEL_ALIAS,
    )

    # ========================================================
    # 4. LOAD DEPLOYED MODEL
    # ========================================================

    model_deployment_service = (
        prediction_service_loader(
            pipeline_name=pipeline_name,
            pipeline_step_name=(
                pipeline_step_name
            ),
            model_name=(
                DEPLOYMENT_MODEL_NAME
            ),
        )
    )

    # ========================================================
    # 5. PREDICTION
    # ========================================================

    predictor(
        service=model_deployment_service,
        data=current_data,
    )