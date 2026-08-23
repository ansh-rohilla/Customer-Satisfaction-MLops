import json
import logging
from typing import Dict, Tuple

import mlflow
import numpy as np
import pandas as pd

from mlflow import MlflowClient
from zenml import step


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


# ============================================================
# PSI CALCULATION
# ============================================================

def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = PSI_BINS,
) -> float:
    """
    Calculate Population Stability Index.

    PSI < 0.10:
        Low / no significant drift

    0.10 <= PSI < 0.25:
        Moderate drift

    PSI >= 0.25:
        Significant drift
    """

    reference = np.asarray(
        reference,
        dtype=float,
    )

    current = np.asarray(
        current,
        dtype=float,
    )

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
        return 0.0

    # ========================================================
    # CONSTANT REFERENCE FEATURE
    # ========================================================

    if np.min(reference) == np.max(reference):

        if np.all(
            current == reference[0]
        ):
            return 0.0

        return 1.0

    # ========================================================
    # QUANTILE BINNING
    # ========================================================

    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    breakpoints = np.quantile(
        reference,
        quantiles,
    )

    breakpoints = np.unique(
        breakpoints
    )

    if len(breakpoints) < 2:
        return 0.0

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

    # ========================================================
    # CONVERT TO PROPORTIONS
    # ========================================================

    reference_percent = (
        reference_counts / len(reference)
    )

    current_percent = (
        current_counts / len(current)
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
    Retrieve the model assigned to the specified MLflow
    alias and download its reference profile.

    Returns:

        reference_profile
        run_id
    """

    client = MlflowClient()

    # ========================================================
    # GET MODEL VERSION
    # ========================================================

    model_version = (
        client.get_model_version_by_alias(
            model_name,
            alias,
        )
    )

    run_id = model_version.run_id

    if not run_id:

        raise RuntimeError(
            "The registered model version "
            "does not contain an MLflow run ID."
        )

    print(
        f"MLflow model      : {model_name}"
    )

    print(
        f"Model alias       : {alias}"
    )

    print(
        f"Model version     : "
        f"{model_version.version}"
    )

    print(
        f"Training run ID   : {run_id}"
    )

    # ========================================================
    # DOWNLOAD REFERENCE PROFILE
    # ========================================================

    local_path = client.download_artifacts(
        run_id=run_id,
        path=REFERENCE_PROFILE_ARTIFACT,
    )

    print(
        "Reference profile downloaded:"
    )

    print(local_path)

    # ========================================================
    # LOAD JSON
    # ========================================================

    with open(
        local_path,
        "r",
        encoding="utf-8",
    ) as file:

        reference_profile = json.load(
            file
        )

    if not isinstance(
        reference_profile,
        dict,
    ):

        raise ValueError(
            "Reference profile must "
            "contain a JSON object."
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
    Convert inference data to numeric values.
    """

    current_data = data.copy()

    for column in current_data.columns:

        current_data[column] = pd.to_numeric(
            current_data[column],
            errors="coerce",
        )

    return current_data


# ============================================================
# RECONSTRUCT REFERENCE DISTRIBUTION
# ============================================================

def reconstruct_reference_distribution(
    profile: Dict,
    samples: int = 10000,
) -> np.ndarray:

    """
    Reconstruct an approximate reference distribution from
    the statistical profile generated by model_train.py.

    The profile contains:

        min
        q01
        q05
        q25
        median
        q75
        q95
        q99
        max

    This approximation is sufficient for monitoring when
    raw training observations are not stored.
    """

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

    for key in required_keys:

        if key not in profile:

            raise ValueError(
                f"Reference profile is missing "
                f"required key '{key}'."
            )

    # ========================================================
    # QUANTILE POINTS
    # ========================================================

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

    # ========================================================
    # REMOVE INVALID VALUES
    # ========================================================

    valid_mask = np.isfinite(
        quantile_values
    )

    quantile_values = (
        quantile_values[valid_mask]
    )

    quantile_points = (
        quantile_points[valid_mask]
    )

    if len(quantile_values) < 2:

        return quantile_values

    # ========================================================
    # REMOVE DUPLICATE QUANTILES
    # ========================================================

    unique_values, unique_indices = (
        np.unique(
            quantile_values,
            return_index=True,
        )
    )

    unique_points = (
        quantile_points[unique_indices]
    )

    if len(unique_values) < 2:

        return unique_values

    # ========================================================
    # GENERATE UNIFORM PROBABILITY SAMPLES
    # ========================================================

    probabilities = np.linspace(
        0,
        1,
        samples,
    )

    reconstructed = np.interp(
        probabilities,
        unique_points,
        unique_values,
    )

    return reconstructed


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

        True  -> significant drift detected
        False -> no significant drift
    """

    try:

        # ====================================================
        # START
        # ====================================================

        print("\n" + "=" * 60)
        print("STARTING DATA DRIFT MONITORING")
        print("=" * 60)

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
            prepare_current_data(data)
        )

        print(
            f"Reference features : "
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

                drift_results.append(
                    {
                        "feature": feature,
                        "psi": None,
                        "status": "MISSING",
                        "reference_count": int(
                            profile.get(
                                "count",
                                0,
                            )
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

                logging.warning(
                    f"Could not reconstruct "
                    f"reference distribution "
                    f"for {feature}: {exc}"
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
                        "reference_count": len(
                            reference_array
                        ),
                        "current_count": 0,
                    }
                )

                continue

            # ------------------------------------------------
            # PSI
            # ------------------------------------------------

            psi = calculate_psi(
                reference=reference_array,
                current=current_array,
                bins=psi_bins,
            )

            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

            status = classify_psi(
                psi=psi,
                low_threshold=psi_low_threshold,
                medium_threshold=psi_medium_threshold,
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
                    "reference_count": len(
                        reference_array
                    ),
                    "current_count": len(
                        current_array
                    ),
                }
            )

            print(
                f"{feature:<35}"
                f" PSI={psi:.4f}"
                f" | {status}"
            )

            # ------------------------------------------------
            # MLFLOW METRIC
            # ------------------------------------------------

            mlflow.log_metric(
                f"drift_psi_{feature}",
                float(psi),
            )

        # ====================================================
        # OVERALL STATISTICS
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
            drift_results
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
            "psi_thresholds": {
                "low": psi_low_threshold,
                "medium": psi_medium_threshold,
            },
            "psi_bins": psi_bins,
            "total_features": total_features,
            "low_drift_features": low_count,
            "medium_drift_features": medium_count,
            "high_drift_features": high_count,
            "missing_features": missing_count,
            "mean_psi": mean_psi,
            "max_psi": max_psi,
            "drift_detected": drift_detected,
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

        print("\n" + "=" * 60)
        print("DRIFT MONITORING RESULTS")
        print("=" * 60)

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
            f"Drift Detected : "
            f"{'YES' if drift_detected else 'NO'}"
        )

        if high_drift_features:

            print(
                "\nHigh Drift Features:"
            )

            for feature in high_drift_features:

                print(
                    f"  - {feature}"
                )

        if missing_features:

            print(
                "\nMissing Features:"
            )

            for feature in missing_features:

                print(
                    f"  - {feature}"
                )

        print("=" * 60)

        return drift_detected

    except Exception as exc:

        logging.error(
            f"Drift monitoring failed: {exc}"
        )

        raise