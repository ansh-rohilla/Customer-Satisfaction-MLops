import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from zenml import step


# ============================================================
# PROJECT / ENVIRONMENT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"

# IMPORTANT:
# ZenML may already define MLFLOW_TRACKING_URI in the
# step environment. override=True ensures the project's
# .env value is used instead.
load_dotenv(
    ENV_FILE,
    override=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

REGISTERED_MODEL_NAME = "Customer-Satisfaction-Model"

MODEL_ALIAS = "champion"

REFERENCE_PROFILE_ARTIFACT = (
    "monitoring/reference_profile.json"
)

PSI_BINS = 10

PSI_LOW_THRESHOLD = 0.10

PSI_MEDIUM_THRESHOLD = 0.25

REFERENCE_RECONSTRUCTION_SAMPLES = 10000


# ============================================================
# MLFLOW TRACKING URI
# ============================================================

def resolve_mlflow_tracking_uri() -> str:
    """
    Resolve the MLflow tracking URI from the project .env.

    The drift-monitoring step must use exactly the same
    MLflow backend as the training/registration pipeline.
    """

    # Reload explicitly with override=True in case ZenML
    # modified the environment before executing this step.
    load_dotenv(
        ENV_FILE,
        override=True,
    )

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI"
    )

    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not configured.\n\n"
            f"Expected .env file:\n{ENV_FILE}\n\n"
            "Please define MLFLOW_TRACKING_URI in .env."
        )

    # Explicitly configure MLflow for this process.
    mlflow.set_tracking_uri(
        tracking_uri
    )

    resolved_uri = (
        mlflow.get_tracking_uri()
    )

    print(
        "\nMLflow tracking URI resolved:"
    )

    print(
        resolved_uri
    )

    # Diagnostic information.
    print(
        "\nMLflow environment variable:"
    )

    print(
        os.environ.get(
            "MLFLOW_TRACKING_URI"
        )
    )

    return resolved_uri


# ============================================================
# CREATE MLFLOW CLIENT
# ============================================================

def get_mlflow_client() -> Tuple[MlflowClient, str]:
    """
    Create an MLflow client using the project's
    configured MLflow backend.
    """

    tracking_uri = (
        resolve_mlflow_tracking_uri()
    )

    # Explicitly set again so every MLflow call in this
    # process uses the same backend.
    mlflow.set_tracking_uri(
        tracking_uri
    )

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "MLFLOW DRIFT MONITOR CONFIGURATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Tracking URI      : "
        f"{tracking_uri}"
    )

    print(
        f"Registered Model  : "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Model Alias       : "
        f"{MODEL_ALIAS}"
    )

    print(
        f"Environment File  : "
        f"{ENV_FILE}"
    )

    print(
        "=" * 60
    )

    return client, tracking_uri

# ============================================================
# VERIFY REGISTERED MODEL
# ============================================================

def verify_registered_model(
    client: MlflowClient,
    model_name: str,
) -> None:
    """
    Verify that the registered model exists.
    """

    try:

        registered_model = (
            client.get_registered_model(
                model_name
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "\nMLflow registered model could not "
            "be found.\n\n"
            f"Model name   : {model_name}\n"
            f"Tracking URI : {client.tracking_uri}\n\n"
            "Make sure the drift monitor is using "
            "the same MLflow backend as the training "
            "pipeline."
        ) from exc

    print(
        "\nRegistered model found:"
    )

    print(
        f"Model : {registered_model.name}"
    )


# ============================================================
# PSI CALCULATION
# ============================================================

def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = PSI_BINS,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI < 0.10:
        LOW / no significant drift

    0.10 <= PSI < 0.25:
        MEDIUM / moderate drift

    PSI >= 0.25:
        HIGH / significant drift
    """

    if bins < 2:
        raise ValueError(
            "PSI bins must be at least 2."
        )

    reference = np.asarray(
        reference,
        dtype=float,
    )

    current = np.asarray(
        current,
        dtype=float,
    )

    # Remove NaN and infinity.
    reference = reference[
        np.isfinite(reference)
    ]

    current = current[
        np.isfinite(current)
    ]

    if (
        len(reference) == 0
        or len(current) == 0
    ):
        raise ValueError(
            "Reference or current distribution "
            "contains no valid numeric observations."
        )

    # ========================================================
    # CONSTANT REFERENCE FEATURE
    # ========================================================

    reference_min = np.min(
        reference
    )

    reference_max = np.max(
        reference
    )

    if reference_min == reference_max:

        if np.all(
            current == reference_min
        ):
            return 0.0

        return 1.0

    # ========================================================
    # QUANTILE BINNING
    # ========================================================

    quantiles = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    breakpoints = np.quantile(
        reference,
        quantiles,
    )

    breakpoints = np.unique(
        breakpoints
    )

    # If the reference contains too few unique values.
    if len(breakpoints) < 2:

        if np.all(
            current == reference[0]
        ):
            return 0.0

        return 1.0

    # ========================================================
    # HISTOGRAMS
    # ========================================================

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints,
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints,
    )

    reference_counts = (
        reference_counts.astype(float)
    )

    current_counts = (
        current_counts.astype(float)
    )

    # ========================================================
    # VALUES OUTSIDE REFERENCE RANGE
    # ========================================================

    current_below = np.sum(
        current < breakpoints[0]
    )

    current_above = np.sum(
        current > breakpoints[-1]
    )

    if current_below > 0:

        current_counts[0] += (
            current_below
        )

    if current_above > 0:

        current_counts[-1] += (
            current_above
        )

    # ========================================================
    # CONVERT TO PROPORTIONS
    # ========================================================

    reference_percent = (
        reference_counts
        / len(reference)
    )

    current_percent = (
        current_counts
        / len(current)
    )

    # ========================================================
    # AVOID LOG(0)
    # ========================================================

    epsilon = 1e-6

    reference_percent = np.clip(
        reference_percent,
        epsilon,
        None,
    )

    current_percent = np.clip(
        current_percent,
        epsilon,
        None,
    )

    # Normalize after clipping.
    reference_percent = (
        reference_percent
        / np.sum(reference_percent)
    )

    current_percent = (
        current_percent
        / np.sum(current_percent)
    )

    # ========================================================
    # PSI
    # ========================================================

    psi = np.sum(
        (
            current_percent
            - reference_percent
        )
        * np.log(
            current_percent
            / reference_percent
        )
    )

    return float(psi)


# ============================================================
# PSI CLASSIFICATION
# ============================================================

def classify_psi(
    psi: float,
    low_threshold: float = PSI_LOW_THRESHOLD,
    medium_threshold: float = PSI_MEDIUM_THRESHOLD,
) -> str:
    """
    Classify PSI into LOW, MEDIUM, or HIGH.
    """

    if psi < low_threshold:
        return "LOW"

    if psi < medium_threshold:
        return "MEDIUM"

    return "HIGH"


# ============================================================
# LOAD REFERENCE PROFILE FROM MLFLOW
# ============================================================

def load_reference_profile_from_mlflow(
    model_name: str,
    alias: str,
) -> Tuple[Dict, str]:
    """
    Retrieve the model version assigned to the specified
    MLflow alias and download its reference profile artifact.

    Returns:

        reference_profile
        run_id
    """

    # ========================================================
    # CREATE CLIENT
    # ========================================================

    client, tracking_uri = (
        get_mlflow_client()
    )

    # ========================================================
    # VERIFY REGISTERED MODEL
    # ========================================================

    print(
        "\nVerifying MLflow registered model..."
    )

    verify_registered_model(
        client=client,
        model_name=model_name,
    )

    # ========================================================
    # GET MODEL VERSION BY ALIAS
    # ========================================================

    try:

        model_version = (
            client.get_model_version_by_alias(
                model_name,
                alias,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "\nCould not find MLflow model alias.\n\n"
            f"Model        : {model_name}\n"
            f"Alias        : {alias}\n"
            f"Tracking URI : {tracking_uri}\n\n"
            "Make sure the registered model has "
            f"the '{alias}' alias."
        ) from exc

    # ========================================================
    # RUN ID
    # ========================================================

    run_id = model_version.run_id

    if not run_id:

        raise RuntimeError(
            "The registered model version does not "
            "contain an MLflow run ID."
        )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    print(
        f"\nMLflow model      : "
        f"{model_name}"
    )

    print(
        f"Model alias       : "
        f"{alias}"
    )

    print(
        f"Model version     : "
        f"{model_version.version}"
    )

    print(
        f"Training run ID   : "
        f"{run_id}"
    )

    print(
        f"Tracking URI      : "
        f"{tracking_uri}"
    )

    # ========================================================
    # DOWNLOAD REFERENCE PROFILE
    # ========================================================

    try:

        local_path = (
            client.download_artifacts(
                run_id=run_id,
                path=REFERENCE_PROFILE_ARTIFACT,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "\nCould not download reference profile "
            "from MLflow.\n\n"
            f"Run ID     : {run_id}\n"
            f"Artifact   : "
            f"{REFERENCE_PROFILE_ARTIFACT}\n"
            f"Tracking URI : {tracking_uri}\n\n"
            "Make sure the training run logged the "
            "reference profile at the expected path."
        ) from exc

    print(
        "\nReference profile downloaded:"
    )

    print(
        local_path
    )

    # ========================================================
    # LOAD JSON
    # ========================================================

    try:

        with open(
            local_path,
            "r",
            encoding="utf-8",
        ) as file:

            reference_profile = (
                json.load(file)
            )

    except Exception as exc:

        raise RuntimeError(
            "Could not read the reference "
            "profile JSON."
        ) from exc

    if not isinstance(
        reference_profile,
        dict,
    ):

        raise ValueError(
            "Reference profile must contain "
            "a JSON object."
        )

    return (
        reference_profile,
        run_id,
    )


# ============================================================
# PREPARE CURRENT DATA
# ============================================================

def prepare_current_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert inference data columns to numeric values.

    Non-numeric values are converted to NaN.
    """

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Drift monitoring data must be "
            "a pandas DataFrame."
        )

    current_data = data.copy()

    for column in current_data.columns:

        current_data[column] = (
            pd.to_numeric(
                current_data[column],
                errors="coerce",
            )
        )

    return current_data


# ============================================================
# RECONSTRUCT REFERENCE DISTRIBUTION
# ============================================================

def reconstruct_reference_distribution(
    profile: Any,
    samples: int = REFERENCE_RECONSTRUCTION_SAMPLES,
) -> np.ndarray:
    """
    Reconstruct an approximate reference distribution.

    Supported formats:

    1. Legacy raw list

    2. Histogram profile

        {
            "min": ...,
            "max": ...,
            "histogram_bins": [...],
            "histogram_counts": [...]
        }

    3. Quantile profile

        {
            "min": ...,
            "q01": ...,
            "q05": ...,
            "q25": ...,
            "median": ...,
            "q75": ...,
            "q95": ...,
            "q99": ...,
            "max": ...
        }

    4. Simple statistical profile

        {
            "min": ...,
            "max": ...,
            "mean": ...,
            "std": ...
        }
    """

    if samples <= 0:
        raise ValueError(
            "Number of reconstruction samples "
            "must be greater than zero."
        )

    # ========================================================
    # LEGACY RAW-LIST FORMAT
    # ========================================================

    if isinstance(
        profile,
        list,
    ):

        values = np.asarray(
            profile,
            dtype=float,
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:

            raise ValueError(
                "Legacy reference profile "
                "contains no valid values."
            )

        return values

    # ========================================================
    # PROFILE MUST BE DICT
    # ========================================================

    if not isinstance(
        profile,
        dict,
    ):

        raise ValueError(
            "Invalid reference profile format."
        )

    # ========================================================
    # HISTOGRAM-BASED PROFILE
    # ========================================================

    histogram_bins = profile.get(
        "histogram_bins"
    )

    histogram_counts = profile.get(
        "histogram_counts"
    )

    if (
        histogram_bins is not None
        and histogram_counts is not None
    ):

        bins = np.asarray(
            histogram_bins,
            dtype=float,
        )

        counts = np.asarray(
            histogram_counts,
            dtype=int,
        )

        if (
            len(bins) >= 2
            and len(counts) == len(bins) - 1
            and np.sum(counts) > 0
            and np.all(np.isfinite(bins))
            and np.all(np.isfinite(counts))
        ):

            rng = np.random.default_rng(
                42
            )

            reconstructed_parts = []

            total_count = int(
                np.sum(counts)
            )

            for index, count in enumerate(
                counts
            ):

                if count <= 0:
                    continue

                left = bins[index]

                right = bins[index + 1]

                generated_size = int(
                    round(
                        samples
                        * (
                            count
                            / total_count
                        )
                    )
                )

                if generated_size <= 0:
                    continue

                # Protect against invalid bin ranges.
                if right < left:
                    continue

                if right == left:

                    generated = np.full(
                        generated_size,
                        left,
                        dtype=float,
                    )

                else:

                    generated = (
                        rng.uniform(
                            left,
                            right,
                            size=generated_size,
                        )
                    )

                reconstructed_parts.append(
                    generated
                )

            if reconstructed_parts:

                reconstructed = (
                    np.concatenate(
                        reconstructed_parts
                    )
                )

                if len(reconstructed) > 0:
                    return reconstructed

    # ========================================================
    # QUANTILE PROFILE
    # ========================================================

    required_keys = [
        "min",
        "q01",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "q99",
        "max",
    ]

    if all(
        key in profile
        for key in required_keys
    ):

        quantile_values = np.array(
            [
                profile["min"],
                profile["q01"],
                profile["q05"],
                profile["q25"],
                profile["median"],
                profile["q75"],
                profile["q95"],
                profile["q99"],
                profile["max"],
            ],
            dtype=float,
        )

        quantile_points = np.array(
            [
                0.00,
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
                1.00,
            ],
            dtype=float,
        )

        valid_mask = np.isfinite(
            quantile_values
        )

        quantile_values = (
            quantile_values[
                valid_mask
            ]
        )

        quantile_points = (
            quantile_points[
                valid_mask
            ]
        )

        if len(
            quantile_values
        ) == 0:

            raise ValueError(
                "Reference profile contains "
                "no valid quantile values."
            )

        # Ensure quantile values are monotonic.
        quantile_values = (
            np.maximum.accumulate(
                quantile_values
            )
        )

        if len(
            np.unique(
                quantile_values
            )
        ) == 1:

            return np.full(
                samples,
                quantile_values[0],
                dtype=float,
            )

        probabilities = np.linspace(
            0.000001,
            0.999999,
            samples,
        )

        reconstructed = np.interp(
            probabilities,
            quantile_points,
            quantile_values,
        )

        return reconstructed

    # ========================================================
    # SIMPLE STATISTICAL PROFILE
    # ========================================================

    if (
        "min" in profile
        and "max" in profile
    ):

        feature_min = float(
            profile["min"]
        )

        feature_max = float(
            profile["max"]
        )

        if not np.isfinite(
            feature_min
        ) or not np.isfinite(
            feature_max
        ):

            raise ValueError(
                "Reference profile contains "
                "invalid min/max values."
            )

        if feature_min > feature_max:

            raise ValueError(
                "Reference profile min cannot "
                "be greater than max."
            )

        if feature_min == feature_max:

            return np.full(
                samples,
                feature_min,
                dtype=float,
            )

        mean = profile.get(
            "mean",
            (
                feature_min
                + feature_max
            ) / 2,
        )

        std = profile.get(
            "std",
            (
                feature_max
                - feature_min
            ) / 6,
        )

        mean = float(mean)

        std = float(std)

        if (
            not np.isfinite(std)
            or std <= 0
        ):

            std = (
                feature_max
                - feature_min
            ) / 6

        rng = np.random.default_rng(
            42
        )

        reconstructed = rng.normal(
            loc=mean,
            scale=std,
            size=samples,
        )

        reconstructed = np.clip(
            reconstructed,
            feature_min,
            feature_max,
        )

        return reconstructed

    # ========================================================
    # INVALID FORMAT
    # ========================================================

    raise ValueError(
        "Reference profile format is not "
        "supported. Expected histogram, "
        "quantile, statistical, or legacy "
        "raw-list format."
    )


# ============================================================
# DRIFT MONITORING STEP
# ============================================================

@step(
    enable_cache=False,
)
def drift_monitor(
    data: pd.DataFrame,
    model_name: str = REGISTERED_MODEL_NAME,
    model_alias: str = MODEL_ALIAS,
    psi_low_threshold: float = PSI_LOW_THRESHOLD,
    psi_medium_threshold: float = PSI_MEDIUM_THRESHOLD,
    psi_bins: int = PSI_BINS,
) -> bool:
    """
    Compare incoming inference data against the training
    reference profile stored with the champion MLflow model.

    Returns:

        True:
            Significant drift detected.

        False:
            No significant drift detected.
    """

    try:

        # ====================================================
        # VALIDATE PARAMETERS
        # ====================================================

        if psi_bins < 2:

            raise ValueError(
                "psi_bins must be at least 2."
            )

        if (
            psi_low_threshold < 0
            or psi_medium_threshold <= psi_low_threshold
        ):

            raise ValueError(
                "PSI thresholds must satisfy:\n"
                "0 <= low_threshold < "
                "medium_threshold."
            )

        # ====================================================
        # START
        # ====================================================

        print(
            "\n" + "=" * 60
        )

        print(
            "STARTING DATA DRIFT MONITORING"
        )

        print(
            "=" * 60
        )

        # ====================================================
        # INITIALIZE MLFLOW
        # ====================================================

        client, tracking_uri = (
            get_mlflow_client()
        )

        print(
            "\nDrift monitor MLflow backend:"
        )

        print(
            tracking_uri
        )

        # ====================================================
        # LOAD REFERENCE PROFILE
        # ====================================================

        (
            reference_profile,
            run_id,
        ) = load_reference_profile_from_mlflow(
            model_name=model_name,
            alias=model_alias,
        )

        # ====================================================
        # PREPARE CURRENT DATA
        # ====================================================

        current_data = (
            prepare_current_data(
                data
            )
        )

        print(
            f"\nReference features : "
            f"{len(reference_profile)}"
        )

        print(
            f"Current features   : "
            f"{len(current_data.columns)}"
        )

        # ====================================================
        # RESULT STORAGE
        # ====================================================

        drift_results = []

        low_drift_features = []

        medium_drift_features = []

        high_drift_features = []

        missing_features = []

        # ====================================================
        # FEATURE-BY-FEATURE PSI
        # ====================================================

        for (
            feature,
            profile,
        ) in reference_profile.items():

            # ------------------------------------------------
            # FEATURE MISSING
            # ------------------------------------------------

            if feature not in current_data.columns:

                print(
                    f"{feature:<35}"
                    f" MISSING"
                )

                if isinstance(
                    profile,
                    dict,
                ):

                    reference_count = (
                        profile.get(
                            "sample_count",
                            profile.get(
                                "count",
                                0,
                            ),
                        )
                    )

                else:

                    try:
                        reference_count = (
                            len(profile)
                        )
                    except TypeError:
                        reference_count = 0

                drift_results.append(
                    {
                        "feature": feature,
                        "psi": None,
                        "status": "MISSING",
                        "reference_count": (
                            reference_count
                        ),
                        "current_count": 0,
                    }
                )

                missing_features.append(
                    feature
                )

                continue

            # ------------------------------------------------
            # RECONSTRUCT REFERENCE DATA
            # ------------------------------------------------

            try:

                reference_array = (
                    reconstruct_reference_distribution(
                        profile
                    )
                )

            except Exception as exc:

                logging.error(
                    "Could not reconstruct "
                    f"reference distribution "
                    f"for {feature}: {exc}"
                )

                drift_results.append(
                    {
                        "feature": feature,
                        "psi": None,
                        "status": "ERROR",
                        "reference_count": 0,
                        "current_count": 0,
                        "error": str(exc),
                    }
                )

                continue

            # ------------------------------------------------
            # CURRENT DATA
            # ------------------------------------------------

            current_array = (
                current_data[feature]
                .dropna()
                .astype(float)
                .to_numpy()
            )

            # ------------------------------------------------
            # EMPTY CURRENT DATA
            # ------------------------------------------------

            if len(current_array) == 0:

                print(
                    f"{feature:<35}"
                    f" NO CURRENT DATA"
                )

                drift_results.append(
                    {
                        "feature": feature,
                        "psi": None,
                        "status": "NO_DATA",
                        "reference_count": (
                            len(
                                reference_array
                            )
                        ),
                        "current_count": 0,
                    }
                )

                continue

            # ------------------------------------------------
            # CALCULATE PSI
            # ------------------------------------------------

            try:

                psi = calculate_psi(
                    reference=reference_array,
                    current=current_array,
                    bins=psi_bins,
                )

            except Exception as exc:

                logging.error(
                    f"PSI calculation failed "
                    f"for {feature}: {exc}"
                )

                drift_results.append(
                    {
                        "feature": feature,
                        "psi": None,
                        "status": "ERROR",
                        "reference_count": (
                            len(
                                reference_array
                            )
                        ),
                        "current_count": (
                            len(
                                current_array
                            )
                        ),
                        "error": str(exc),
                    }
                )

                continue

            # ------------------------------------------------
            # CLASSIFY
            # ------------------------------------------------

            status = classify_psi(
                psi=psi,
                low_threshold=(
                    psi_low_threshold
                ),
                medium_threshold=(
                    psi_medium_threshold
                ),
            )

            if status == "LOW":

                low_drift_features.append(
                    feature
                )

            elif status == "MEDIUM":

                medium_drift_features.append(
                    feature
                )

            else:

                high_drift_features.append(
                    feature
                )

            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            drift_results.append(
                {
                    "feature": feature,
                    "psi": round(
                        psi,
                        6,
                    ),
                    "status": status,
                    "reference_count": (
                        len(
                            reference_array
                        )
                    ),
                    "current_count": (
                        len(
                            current_array
                        )
                    ),
                }
            )

            print(
                f"{feature:<35}"
                f" PSI={psi:.6f}"
                f" | {status}"
            )

            # ------------------------------------------------
            # MLFLOW FEATURE PSI METRIC
            # ------------------------------------------------

            mlflow.log_metric(
                f"drift_psi_{feature}",
                float(psi),
            )

        # ====================================================
        # OVERALL PSI
        # ====================================================

        valid_psi_values = [
            result["psi"]
            for result in drift_results
            if result["psi"] is not None
        ]

        if valid_psi_values:

            mean_psi = float(
                np.mean(
                    valid_psi_values
                )
            )

            max_psi = float(
                np.max(
                    valid_psi_values
                )
            )

        else:

            mean_psi = 0.0

            max_psi = 0.0

        # ====================================================
        # COUNTS
        # ====================================================

        total_features = len(
            reference_profile
        )

        low_count = len(
            low_drift_features
        )

        medium_count = len(
            medium_drift_features
        )

        high_count = len(
            high_drift_features
        )

        missing_count = len(
            missing_features
        )

        error_count = sum(
            1
            for result in drift_results
            if result["status"] == "ERROR"
        )

        no_data_count = sum(
            1
            for result in drift_results
            if result["status"] == "NO_DATA"
        )

        # ====================================================
        # FINAL DRIFT DECISION
        # ====================================================

        drift_detected = (
            high_count > 0
            or missing_count > 0
        )

        # ====================================================
        # MLFLOW METRICS
        # ====================================================

        mlflow.log_metric(
            "drift_mean_psi",
            mean_psi,
        )

        mlflow.log_metric(
            "drift_max_psi",
            max_psi,
        )

        mlflow.log_metric(
            "drift_low_feature_count",
            float(low_count),
        )

        mlflow.log_metric(
            "drift_medium_feature_count",
            float(medium_count),
        )

        mlflow.log_metric(
            "drift_high_feature_count",
            float(high_count),
        )

        mlflow.log_metric(
            "drift_missing_feature_count",
            float(missing_count),
        )

        mlflow.log_metric(
            "drift_error_feature_count",
            float(error_count),
        )

        mlflow.log_metric(
            "drift_no_data_feature_count",
            float(no_data_count),
        )

        mlflow.log_metric(
            "drift_total_feature_count",
            float(total_features),
        )

        # ====================================================
        # MLFLOW PARAMETERS
        # ====================================================

        mlflow.log_param(
            "drift_monitoring_method",
            "PSI",
        )

        mlflow.log_param(
            "drift_model_name",
            model_name,
        )

        mlflow.log_param(
            "drift_model_alias",
            model_alias,
        )

        mlflow.log_param(
            "drift_reference_run_id",
            run_id,
        )

        mlflow.log_param(
            "drift_psi_bins",
            psi_bins,
        )

        mlflow.log_param(
            "drift_low_threshold",
            psi_low_threshold,
        )

        mlflow.log_param(
            "drift_medium_threshold",
            psi_medium_threshold,
        )

        mlflow.log_param(
            "drift_detected",
            drift_detected,
        )

        # ====================================================
        # DRIFT REPORT
        # ====================================================

        report = {

            "monitoring_method": "PSI",

            "model_name": model_name,

            "model_alias": model_alias,

            "reference_run_id": run_id,

            "mlflow_tracking_uri": tracking_uri,

            "psi_thresholds": {
                "low": (
                    psi_low_threshold
                ),
                "medium": (
                    psi_medium_threshold
                ),
            },

            "psi_bins": psi_bins,

            "total_features": (
                total_features
            ),

            "low_drift_features": (
                low_count
            ),

            "medium_drift_features": (
                medium_count
            ),

            "high_drift_features": (
                high_count
            ),

            "missing_features": (
                missing_count
            ),

            "error_features": (
                error_count
            ),

            "no_data_features": (
                no_data_count
            ),

            "mean_psi": mean_psi,

            "max_psi": max_psi,

            "drift_detected": (
                drift_detected
            ),

            "features": drift_results,
        }

        # ====================================================
        # LOG REPORT
        # ====================================================

        mlflow.log_dict(
            report,
            "monitoring/drift_report.json",
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        print(
            "\n" + "=" * 60
        )

        print(
            "DRIFT MONITORING RESULTS"
        )

        print(
            "=" * 60
        )

        print(
            f"Mean PSI       : "
            f"{mean_psi:.6f}"
        )

        print(
            f"Maximum PSI    : "
            f"{max_psi:.6f}"
        )

        print(
            f"Low Drift      : "
            f"{low_count}"
        )

        print(
            f"Medium Drift   : "
            f"{medium_count}"
        )

        print(
            f"High Drift     : "
            f"{high_count}"
        )

        print(
            f"Missing        : "
            f"{missing_count}"
        )

        print(
            f"Errors         : "
            f"{error_count}"
        )

        print(
            f"No Data        : "
            f"{no_data_count}"
        )

        print(
            f"Drift Detected : "
            f"{'YES' if drift_detected else 'NO'}"
        )

        # ====================================================
        # HIGH DRIFT FEATURES
        # ====================================================

        if high_drift_features:

            print(
                "\nHigh Drift Features:"
            )

            for feature in (
                high_drift_features
            ):

                print(
                    f"  - {feature}"
                )

        # ====================================================
        # MEDIUM DRIFT FEATURES
        # ====================================================

        if medium_drift_features:

            print(
                "\nMedium Drift Features:"
            )

            for feature in (
                medium_drift_features
            ):

                print(
                    f"  - {feature}"
                )

        # ====================================================
        # MISSING FEATURES
        # ====================================================

        if missing_features:

            print(
                "\nMissing Features:"
            )

            for feature in (
                missing_features
            ):

                print(
                    f"  - {feature}"
                )

        print(
            "=" * 60
        )

        return drift_detected

    except Exception as exc:

        logging.exception(
            f"Drift monitoring failed: {exc}"
        )

        raise