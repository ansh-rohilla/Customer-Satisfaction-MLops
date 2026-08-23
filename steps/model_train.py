import logging
import os
from pathlib import Path
from typing import Any, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature

from zenml import step
from zenml.client import Client

from model.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
    ModelBenchmark,
    get_candidate_models,
)


# ============================================================
# ENVIRONMENT / PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ============================================================
# MLFLOW MODEL REGISTRY
# ============================================================

REGISTERED_MODEL_NAME = "Customer-Satisfaction-Model"

MODEL_ALIAS = "champion"


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

ENABLE_FINE_TUNING = True

OPTUNA_TRIALS = 10

CV_FOLDS = 5

RANDOM_STATE = 42


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
# DISPLAY CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING CONFIGURATION")
print("=" * 60)

print(
    f"MLflow Tracking URI : "
    f"{mlflow.get_tracking_uri()}"
)

print(
    f"ZenML Experiment    : "
    f"{experiment_tracker.name}"
)

print(
    f"Registered Model    : "
    f"{REGISTERED_MODEL_NAME}"
)

print(
    f"Target Alias        : "
    f"{MODEL_ALIAS}"
)

print(
    f"Fine Tuning         : "
    f"{ENABLE_FINE_TUNING}"
)

print(
    f"Optuna Trials       : "
    f"{OPTUNA_TRIALS}"
)

print(
    f"CV Folds            : "
    f"{CV_FOLDS}"
)

print("=" * 60)


# ============================================================
# HELPER: CREATE MODEL OBJECT
# ============================================================

def _create_model(model_name: str) -> Any:
    """
    Create the appropriate model wrapper based
    on the benchmark-selected model name.
    """

    normalized_name = (
        str(model_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if normalized_name in {
        "lightgbm",
        "light_gbm",
    }:
        return LightGBMModel()

    if normalized_name in {
        "xgboost",
        "xgb",
    }:
        return XGBoostModel()

    if normalized_name in {
        "randomforest",
        "random_forest",
        "random_forest_regressor",
    }:
        return RandomForestModel()

    if normalized_name in {
        "linear_regression",
        "linearregression",
        "linear",
    }:
        return LinearRegressionModel()

    raise ValueError(
        f"Unsupported model selected by benchmark: "
        f"'{model_name}'"
    )


# ============================================================
# HELPER: VALIDATE BENCHMARK RESULTS
# ============================================================

def _validate_benchmark_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate model benchmark results and sort
    models by CV R² in descending order.
    """

    if results is None:
        raise RuntimeError(
            "Model benchmarking returned None."
        )

    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "Model benchmarking must return "
            "a pandas DataFrame."
        )

    if results.empty:
        raise RuntimeError(
            "Model benchmarking returned "
            "an empty DataFrame."
        )

    required_columns = [
        "model",
        "cv_r2",
        "cv_r2_std",
        "cv_rmse",
        "cv_rmse_std",
        "cv_mae",
        "cv_mae_std",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in results.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Benchmark results are missing "
            f"required columns: {missing_columns}"
        )

    metric_columns = [
        "cv_r2",
        "cv_r2_std",
        "cv_rmse",
        "cv_rmse_std",
        "cv_mae",
        "cv_mae_std",
    ]

    results = results.copy()

    for column in metric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

    invalid_mask = (
        results[metric_columns]
        .isnull()
        .any(axis=1)
    )

    if invalid_mask.any():
        invalid_rows = results[invalid_mask]

        raise RuntimeError(
            "Benchmark results contain invalid "
            "numeric values.\n"
            f"Invalid rows:\n{invalid_rows}"
        )

    results = (
        results
        .sort_values(
            by="cv_r2",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return results


# ============================================================
# HELPER: BUILD REFERENCE PROFILE
# ============================================================

def _build_reference_profile(
    x_train: pd.DataFrame,
) -> dict:
    """
    Build statistical reference data from the
    training dataset for drift monitoring.
    """

    if not isinstance(x_train, pd.DataFrame):
        raise TypeError(
            "x_train must be a pandas DataFrame."
        )

    if x_train.empty:
        raise ValueError(
            "x_train is empty."
        )

    reference_profile = {}

    required_profile_keys = [
        "min",
        "q01",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "q99",
        "max",
        "count",
    ]

    for column in x_train.columns:

        numeric_values = (
            pd.to_numeric(
                x_train[column],
                errors="coerce",
            )
            .dropna()
        )

        if numeric_values.empty:
            print(
                f"Skipping reference profile "
                f"for '{column}': "
                f"no numeric values."
            )
            continue

        values = (
            numeric_values
            .astype(float)
            .to_numpy()
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:
            print(
                f"Skipping reference profile "
                f"for '{column}': "
                f"no finite values."
            )
            continue

        quantiles = np.quantile(
            values,
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
        )

        reference_profile[column] = {
            "min": float(quantiles[0]),
            "q01": float(quantiles[1]),
            "q05": float(quantiles[2]),
            "q25": float(quantiles[3]),
            "median": float(quantiles[4]),
            "q75": float(quantiles[5]),
            "q95": float(quantiles[6]),
            "q99": float(quantiles[7]),
            "max": float(quantiles[8]),
            "count": int(len(values)),
        }

    for feature, profile in reference_profile.items():

        missing_keys = [
            key
            for key in required_profile_keys
            if key not in profile
        ]

        if missing_keys:
            raise RuntimeError(
                f"Reference profile for "
                f"feature '{feature}' is missing "
                f"keys: {missing_keys}"
            )

    if not reference_profile:
        raise RuntimeError(
            "Reference profile could not be "
            "created from training data."
        )

    return reference_profile


# ============================================================
# HELPER: FIND EXACT REGISTERED MODEL VERSION
# ============================================================

def _find_registered_model_version(
    client: MlflowClient,
    run_id: str,
) -> str:
    """
    Find the exact MLflow registered model version
    associated with the current training run.
    """

    try:
        model_versions = (
            client.search_model_versions(
                filter_string=(
                    f"name='{REGISTERED_MODEL_NAME}'"
                )
            )
        )

    except Exception as exc:
        raise RuntimeError(
            "Unable to search MLflow Model Registry "
            f"for '{REGISTERED_MODEL_NAME}'."
        ) from exc

    matching_versions = [
        version
        for version in model_versions
        if version.run_id == run_id
    ]

    if not matching_versions:
        raise RuntimeError(
            "No registered model version was found "
            "for the current MLflow run.\n"
            f"Registered Model: "
            f"{REGISTERED_MODEL_NAME}\n"
            f"Run ID: {run_id}\n"
            f"Tracking URI: "
            f"{mlflow.get_tracking_uri()}"
        )

    candidate_version = max(
        matching_versions,
        key=lambda version: int(version.version),
    )

    return str(candidate_version.version)


# ============================================================
# HELPER: VERIFY MLFLOW MODEL REGISTRY
# ============================================================

def _ensure_registered_model_exists(
    client: MlflowClient,
) -> None:
    """
    Make sure the registered model exists.
    """

    try:
        client.get_registered_model(
            REGISTERED_MODEL_NAME
        )

        print(
            f"Registered model already exists: "
            f"{REGISTERED_MODEL_NAME}"
        )

    except Exception:

        print(
            f"Creating registered model: "
            f"{REGISTERED_MODEL_NAME}"
        )

        try:
            client.create_registered_model(
                REGISTERED_MODEL_NAME
            )

        except Exception as exc:

            try:
                client.get_registered_model(
                    REGISTERED_MODEL_NAME
                )

            except Exception:

                raise RuntimeError(
                    "Failed to create or find "
                    f"registered model "
                    f"'{REGISTERED_MODEL_NAME}'."
                ) from exc


# ============================================================
# HELPER: LOG OPTUNA TRIAL HISTORY
# ============================================================

def _log_optuna_trial_history(
    trial_history: list,
) -> None:
    """
    Print and log Optuna trial history to MLflow.
    """

    print("\n" + "=" * 60)
    print("OPTUNA TRIAL SUMMARY")
    print("=" * 60)

    for trial in trial_history:

        trial_number = int(
            trial.get(
                "trial_number",
                0,
            )
        )

        trial_state = trial.get(
            "state",
            "UNKNOWN",
        )

        trial_value = trial.get(
            "value"
        )

        if trial_value is None:

            print(
                f"Trial "
                f"{trial_number + 1:02d} "
                f"| State: "
                f"{trial_state}"
            )

            continue

        trial_r2 = float(
            trial_value
        )

        print(
            f"Trial "
            f"{trial_number + 1:02d} "
            f"| R²: "
            f"{trial_r2:.6f} "
            f"| Params: "
            f"{trial.get('params', {})}"
        )

        mlflow.log_metric(
            f"optuna_trial_"
            f"{trial_number}_r2",
            trial_r2,
        )

        optional_metrics = {
            "cv_r2_std": "r2_std",
            "cv_rmse": "rmse",
            "cv_rmse_std": "rmse_std",
            "cv_mae": "mae",
            "cv_mae_std": "mae_std",
            "duration_seconds": (
                "duration_seconds"
            ),
        }

        for (
            source_key,
            metric_suffix,
        ) in optional_metrics.items():

            value = trial.get(
                source_key
            )

            if value is not None:

                mlflow.log_metric(
                    f"optuna_trial_"
                    f"{trial_number}_"
                    f"{metric_suffix}",
                    float(value),
                )

    print("=" * 60)


# ============================================================
# TRAINING STEP
# ============================================================

@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Any, str]:
    """
    Train, evaluate, tune, register, and validate
    the best customer satisfaction model.

    Returns
    -------
    Tuple[Any, str]
        Trained model and exact registered model version.
    """

    try:

        # ====================================================
        # 1. VALIDATE INPUT DATA
        # ====================================================

        print("\n" + "=" * 60)
        print("VALIDATING TRAINING DATA")
        print("=" * 60)

        if not isinstance(
            x_train,
            pd.DataFrame,
        ):
            raise TypeError(
                "x_train must be a pandas DataFrame."
            )

        if not isinstance(
            x_test,
            pd.DataFrame,
        ):
            raise TypeError(
                "x_test must be a pandas DataFrame."
            )

        if not isinstance(
            y_train,
            (pd.Series, pd.DataFrame),
        ):
            raise TypeError(
                "y_train must be a pandas Series "
                "or DataFrame."
            )

        if not isinstance(
            y_test,
            (pd.Series, pd.DataFrame),
        ):
            raise TypeError(
                "y_test must be a pandas Series "
                "or DataFrame."
            )

        if x_train.empty:
            raise ValueError(
                "x_train is empty."
            )

        if x_test.empty:
            raise ValueError(
                "x_test is empty."
            )

        if len(x_train) != len(y_train):
            raise ValueError(
                "x_train and y_train have "
                "different numbers of rows."
            )

        if len(x_test) != len(y_test):
            raise ValueError(
                "x_test and y_test have "
                "different numbers of rows."
            )

        if list(x_train.columns) != list(
            x_test.columns
        ):
            raise ValueError(
                "Training and test feature columns "
                "do not match.\n"
                f"Train: {list(x_train.columns)}\n"
                f"Test: {list(x_test.columns)}"
            )

        print(
            f"Training rows : {len(x_train)}"
        )

        print(
            f"Test rows     : {len(x_test)}"
        )

        print(
            f"Features      : "
            f"{len(x_train.columns)}"
        )

        print("=" * 60)

        # ====================================================
        # 2. VERIFY MLFLOW
        # ====================================================

        print("\n" + "=" * 60)
        print("VERIFYING MLFLOW")
        print("=" * 60)

        tracking_uri = (
            mlflow.get_tracking_uri()
        )

        print(
            f"Tracking URI: {tracking_uri}"
        )

        client = MlflowClient(
            tracking_uri=tracking_uri
        )

        print(
            "MLflow client initialized successfully."
        )

        print("=" * 60)

        # ====================================================
        # 3. USE ZENML-PROVIDED MLFLOW RUN
        # ====================================================

        print("\n" + "=" * 60)
        print("CHECKING ZENML MLFLOW RUN")
        print("=" * 60)

        active_run = mlflow.active_run()

        if active_run is None:
            raise RuntimeError(
                "No active MLflow run was found. "
                "Make sure the ZenML MLflow experiment "
                "tracker is correctly configured."
            )

        run_id = active_run.info.run_id

        print(
            "Using active ZenML MLflow run."
        )

        print(
            f"Run ID       : {run_id}"
        )

        print(
            f"Experiment ID: "
            f"{active_run.info.experiment_id}"
        )

        print("=" * 60)

        # ====================================================
        # 4. LOG TRAINING CONFIGURATION
        # ====================================================

        mlflow.log_param(
            "training_random_state",
            RANDOM_STATE,
        )

        mlflow.log_param(
            "cv_folds",
            CV_FOLDS,
        )

        mlflow.log_param(
            "fine_tuning_enabled",
            ENABLE_FINE_TUNING,
        )

        mlflow.log_param(
            "registered_model_name",
            REGISTERED_MODEL_NAME,
        )

        mlflow.log_param(
            "model_alias_target",
            MODEL_ALIAS,
        )

        if ENABLE_FINE_TUNING:
            mlflow.log_param(
                "optuna_trials",
                OPTUNA_TRIALS,
            )

        # ====================================================
        # 5. MODEL BENCHMARKING
        # ====================================================

        print("\n" + "=" * 60)
        print("STARTING MODEL BENCHMARKING")
        print("=" * 60)

        candidate_models = (
            get_candidate_models()
        )

        if not candidate_models:
            raise RuntimeError(
                "get_candidate_models() returned "
                "no candidate models."
            )

        benchmark = ModelBenchmark(
            n_splits=CV_FOLDS,
            random_state=RANDOM_STATE,
        )

        results = benchmark.compare_models(
            candidate_models,
            x_train,
            y_train,
        )

        results = _validate_benchmark_results(
            results
        )

        # ====================================================
        # 6. DISPLAY MODEL COMPARISON
        # ====================================================

        print("\n" + "=" * 60)
        print("MODEL COMPARISON RESULTS")
        print("=" * 60)

        display_columns = [
            "model",
            "cv_r2",
            "cv_r2_std",
            "cv_rmse",
            "cv_rmse_std",
            "cv_mae",
            "cv_mae_std",
        ]

        print(
            results[
                display_columns
            ].to_string(index=False)
        )

        print("=" * 60)

        # ====================================================
        # 7. SELECT BEST MODEL
        # ====================================================

        best_model_name = str(
            results.iloc[0]["model"]
        )

        baseline_cv_r2 = float(
            results.iloc[0]["cv_r2"]
        )

        print("\n" + "=" * 60)

        print(
            f"BEST MODEL SELECTED: "
            f"{best_model_name}"
        )

        print(
            f"BASELINE CV R²: "
            f"{baseline_cv_r2:.6f}"
        )

        print("=" * 60)

        # ====================================================
        # 8. LOG BENCHMARK RESULTS
        # ====================================================

        for _, row in results.iterrows():

            model_name = str(
                row["model"]
            )

            safe_model_name = (
                model_name
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            mlflow.log_metric(
                f"{safe_model_name}_cv_r2",
                float(row["cv_r2"]),
            )

            mlflow.log_metric(
                f"{safe_model_name}_cv_r2_std",
                float(row["cv_r2_std"]),
            )

            mlflow.log_metric(
                f"{safe_model_name}_cv_rmse",
                float(row["cv_rmse"]),
            )

            mlflow.log_metric(
                f"{safe_model_name}_cv_rmse_std",
                float(row["cv_rmse_std"]),
            )

            mlflow.log_metric(
                f"{safe_model_name}_cv_mae",
                float(row["cv_mae"]),
            )

            mlflow.log_metric(
                f"{safe_model_name}_cv_mae_std",
                float(row["cv_mae_std"]),
            )

        mlflow.log_param(
            "selected_model",
            best_model_name,
        )

        # ====================================================
        # 9. CREATE SELECTED MODEL
        # ====================================================

        model = _create_model(
            best_model_name
        )

        # ====================================================
        # 10. DEFAULT CV VALUES
        # ====================================================

        best_params = {}

        tuned_cv_r2 = baseline_cv_r2

        tuned_cv_r2_std = float(
            results.iloc[0]["cv_r2_std"]
        )

        tuned_cv_rmse = float(
            results.iloc[0]["cv_rmse"]
        )

        tuned_cv_rmse_std = float(
            results.iloc[0]["cv_rmse_std"]
        )

        tuned_cv_mae = float(
            results.iloc[0]["cv_mae"]
        )

        tuned_cv_mae_std = float(
            results.iloc[0]["cv_mae_std"]
        )

        cv_r2_improvement = 0.0

        cv_r2_improvement_pct = 0.0

        # ====================================================
        # 11. OPTUNA HYPERPARAMETER TUNING
        # ====================================================

        if ENABLE_FINE_TUNING:

            print("\n" + "=" * 60)

            print(
                f"STARTING OPTUNA TUNING "
                f"FOR {best_model_name}"
            )

            print("=" * 60)

            tuner = HyperparameterTuner(
                model=model,
                x_train=x_train,
                y_train=y_train,
                n_splits=CV_FOLDS,
                random_state=RANDOM_STATE,
            )

            tuning_results = tuner.optimize(
                n_trials=OPTUNA_TRIALS,
            )

            if tuning_results is None:
                raise RuntimeError(
                    "Hyperparameter tuning returned None."
                )

            best_params = tuning_results.get(
                "best_params",
                {},
            )

            if not isinstance(
                best_params,
                dict,
            ):
                raise TypeError(
                    "Optuna best_params must be a dictionary."
                )

            tuned_cv_r2 = float(
                tuning_results["best_cv_r2"]
            )

            tuned_cv_r2_std = float(
                tuning_results["cv_r2_std"]
            )

            tuned_cv_rmse = float(
                tuning_results["cv_rmse"]
            )

            tuned_cv_rmse_std = float(
                tuning_results["cv_rmse_std"]
            )

            tuned_cv_mae = float(
                tuning_results["cv_mae"]
            )

            tuned_cv_mae_std = float(
                tuning_results["cv_mae_std"]
            )

            trial_history = (
                tuning_results.get(
                    "trial_history",
                    [],
                )
            )

            best_trial_number = int(
                tuning_results.get(
                    "best_trial_number",
                    0,
                )
            )

            optuna_best_value = float(
                tuning_results.get(
                    "optuna_best_value",
                    tuned_cv_r2,
                )
            )

            cv_r2_improvement = (
                tuned_cv_r2
                - baseline_cv_r2
            )

            if baseline_cv_r2 != 0:

                cv_r2_improvement_pct = (
                    cv_r2_improvement
                    / abs(baseline_cv_r2)
                ) * 100.0

            print("\n" + "=" * 60)
            print("OPTUNA RESULTS")
            print("=" * 60)

            print(
                f"Baseline CV R² : "
                f"{baseline_cv_r2:.6f}"
            )

            print(
                f"Tuned CV R²    : "
                f"{tuned_cv_r2:.6f}"
            )

            print(
                f"CV R² Std      : "
                f"{tuned_cv_r2_std:.6f}"
            )

            print(
                f"CV RMSE        : "
                f"{tuned_cv_rmse:.6f}"
            )

            print(
                f"CV RMSE Std    : "
                f"{tuned_cv_rmse_std:.6f}"
            )

            print(
                f"CV MAE         : "
                f"{tuned_cv_mae:.6f}"
            )

            print(
                f"CV MAE Std     : "
                f"{tuned_cv_mae_std:.6f}"
            )

            print(
                f"R² Improvement : "
                f"{cv_r2_improvement:.6f}"
            )

            print(
                f"Improvement %  : "
                f"{cv_r2_improvement_pct:.2f}%"
            )

            print(
                f"Best Trial     : "
                f"{best_trial_number + 1}"
            )

            print(
                f"Best Parameters: "
                f"{best_params}"
            )

            print("=" * 60)

            # ------------------------------------------------
            # LOG OPTUNA CONFIGURATION
            # ------------------------------------------------

            mlflow.log_param(
                "optuna_enabled",
                True,
            )

            mlflow.log_param(
                "optuna_trials",
                OPTUNA_TRIALS,
            )

            mlflow.log_param(
                "optuna_cv_folds",
                CV_FOLDS,
            )

            mlflow.log_param(
                "optuna_best_trial_number",
                best_trial_number,
            )

            if "sampler" in tuning_results:

                mlflow.log_param(
                    "optuna_sampler",
                    tuning_results["sampler"],
                )

            if "direction" in tuning_results:

                mlflow.log_param(
                    "optuna_direction",
                    tuning_results["direction"],
                )

            # ------------------------------------------------
            # LOG OPTUNA METRICS
            # ------------------------------------------------

            mlflow.log_metric(
                "baseline_cv_r2",
                baseline_cv_r2,
            )

            mlflow.log_metric(
                "tuned_cv_r2",
                tuned_cv_r2,
            )

            mlflow.log_metric(
                "tuned_cv_r2_std",
                tuned_cv_r2_std,
            )

            mlflow.log_metric(
                "tuned_cv_rmse",
                tuned_cv_rmse,
            )

            mlflow.log_metric(
                "tuned_cv_rmse_std",
                tuned_cv_rmse_std,
            )

            mlflow.log_metric(
                "tuned_cv_mae",
                tuned_cv_mae,
            )

            mlflow.log_metric(
                "tuned_cv_mae_std",
                tuned_cv_mae_std,
            )

            mlflow.log_metric(
                "cv_r2_improvement",
                cv_r2_improvement,
            )

            mlflow.log_metric(
                "cv_r2_improvement_pct",
                cv_r2_improvement_pct,
            )

            mlflow.log_metric(
                "optuna_best_trial_r2",
                optuna_best_value,
            )

            # ------------------------------------------------
            # LOG BEST PARAMETERS
            # ------------------------------------------------

            for (
                param_name,
                param_value,
            ) in best_params.items():

                mlflow.log_param(
                    f"tuned_{param_name}",
                    param_value,
                )

            # ------------------------------------------------
            # LOG TRIAL HISTORY
            # ------------------------------------------------

            _log_optuna_trial_history(
                trial_history
            )

        else:

            mlflow.log_param(
                "optuna_enabled",
                False,
            )

        # ====================================================
        # 12. TRAIN FINAL MODEL
        # ====================================================

        print("\n" + "=" * 60)
        print("TRAINING FINAL MODEL")
        print("=" * 60)

        print(
            f"Model: {best_model_name}"
        )

        if ENABLE_FINE_TUNING:

            print(
                "Using optimized hyperparameters."
            )

            trained_model = model.train(
                x_train,
                y_train,
                **best_params,
            )

        else:

            print(
                "Using default model parameters."
            )

            trained_model = model.train(
                x_train,
                y_train,
            )

        if trained_model is None:
            raise RuntimeError(
                "Final model training returned None."
            )

        print(
            "Final model training completed."
        )

        print("=" * 60)

        # ====================================================
        # 13. VERIFY MODEL CAN PREDICT
        # ====================================================

        print("\n" + "=" * 60)
        print("VERIFYING TRAINED MODEL")
        print("=" * 60)

        try:

            test_predictions = (
                trained_model.predict(
                    x_test
                )
            )

        except Exception as exc:

            raise RuntimeError(
                "The trained model could not "
                "generate predictions on x_test."
            ) from exc

        test_predictions = np.asarray(
            test_predictions
        ).reshape(-1)

        if len(test_predictions) != len(
            x_test
        ):
            raise RuntimeError(
                "Prediction count does not match "
                "test dataset size.\n"
                f"Expected: {len(x_test)}\n"
                f"Received: "
                f"{len(test_predictions)}"
            )

        if not np.isfinite(
            test_predictions
        ).all():
            raise RuntimeError(
                "Trained model produced "
                "non-finite predictions."
            )

        print(
            f"Prediction count: "
            f"{len(test_predictions)}"
        )

        print(
            "Model prediction verification passed."
        )

        print("=" * 60)

        # ====================================================
        # 14. LOG FINAL MODEL METADATA
        # ====================================================

        mlflow.set_tag(
            "model_name",
            str(best_model_name),
        )

        mlflow.set_tag(
            "model_registry_name",
            REGISTERED_MODEL_NAME,
        )

        mlflow.set_tag(
            "model_lifecycle",
            "candidate",
        )

        mlflow.set_tag(
            "training_pipeline",
            "continuous_deployment_pipeline",
        )

        mlflow.set_tag(
            "candidate_status",
            "awaiting_quality_gate",
        )

        mlflow.log_metric(
            "final_cv_r2",
            float(tuned_cv_r2),
        )

        mlflow.log_metric(
            "final_cv_rmse",
            float(tuned_cv_rmse),
        )

        mlflow.log_metric(
            "final_cv_mae",
            float(tuned_cv_mae),
        )

        # ====================================================
        # 15. BUILD REFERENCE PROFILE
        # ====================================================

        print("\n" + "=" * 60)
        print("CREATING TRAINING REFERENCE PROFILE")
        print("=" * 60)

        reference_profile = (
            _build_reference_profile(
                x_train
            )
        )

        mlflow.log_dict(
            reference_profile,
            "monitoring/reference_profile.json",
        )

        print(
            "Reference profile logged:"
        )

        print(
            "monitoring/reference_profile.json"
        )

        print(
            f"Reference features: "
            f"{len(reference_profile)}"
        )

        for (
            feature,
            profile,
        ) in reference_profile.items():

            print(
                f"  {feature:<35}"
                f" samples={profile['count']}"
                f" min={profile['min']:.4f}"
                f" median={profile['median']:.4f}"
                f" max={profile['max']:.4f}"
            )

        print("=" * 60)

        # ====================================================
        # 16. VERIFY ACTIVE MLFLOW RUN
        # ====================================================

        active_run = mlflow.active_run()

        if active_run is None:
            raise RuntimeError(
                "No active MLflow run exists before "
                "model registration."
            )

        run_id = active_run.info.run_id

        print("\n" + "=" * 60)
        print("MLFLOW RUN INFORMATION")
        print("=" * 60)

        print(
            f"Run ID      : {run_id}"
        )

        print(
            f"Experiment  : "
            f"{active_run.info.experiment_id}"
        )

        print(
            f"Tracking URI: "
            f"{mlflow.get_tracking_uri()}"
        )

        print("=" * 60)

        # ====================================================
        # 17. ENSURE REGISTERED MODEL EXISTS
        # ====================================================

        print("\n" + "=" * 60)
        print("VERIFYING MLFLOW MODEL REGISTRY")
        print("=" * 60)

        _ensure_registered_model_exists(
            client
        )

        print("=" * 60)

        # ====================================================
        # 18. INFER MODEL SIGNATURE
        # ====================================================

        print("\n" + "=" * 60)
        print("CREATING MODEL SIGNATURE")
        print("=" * 60)

        try:

            train_pred = (
                trained_model.predict(
                    x_train
                )
            )

            train_pred = np.asarray(
                train_pred
            ).reshape(-1)

            signature = infer_signature(
                x_train,
                train_pred,
            )

            print(
                "MLflow model signature created."
            )

        except Exception as exc:

            print(
                "Warning: Could not infer model "
                f"signature: {exc}"
            )

            signature = None

        print("=" * 60)

        # ====================================================
        # 19. LOG + REGISTER FINAL MODEL
        # ====================================================

        print("\n" + "=" * 60)
        print("REGISTERING FINAL MODEL")
        print("=" * 60)

        try:

            model_info = (
                mlflow.sklearn.log_model(
                    sk_model=trained_model,
                    artifact_path="model",
                    registered_model_name=(
                        REGISTERED_MODEL_NAME
                    ),
                    signature=signature,
                    input_example=x_train.head(3),
                    await_registration_for=300,
                )
            )

        except Exception as exc:

            raise RuntimeError(
                "Failed to register the final model "
                "in MLflow Model Registry."
            ) from exc

        print(
            "Model successfully logged "
            "and registered."
        )

        if hasattr(model_info, "model_uri"):
            print(
                f"Model URI: "
                f"{model_info.model_uri}"
            )

        print("=" * 60)

        # ====================================================
        # 20. FIND EXACT CANDIDATE VERSION
        # ====================================================

        print("\n" + "=" * 60)
        print("FINDING EXACT REGISTERED MODEL VERSION")
        print("=" * 60)

        candidate_version_number = (
            _find_registered_model_version(
                client=client,
                run_id=run_id,
            )
        )

        print(
            f"Registered Model : "
            f"{REGISTERED_MODEL_NAME}"
        )

        print(
            f"Candidate Version: "
            f"{candidate_version_number}"
        )

        print(
            f"Training Run ID  : "
            f"{run_id}"
        )

        print(
            "Lifecycle        : candidate"
        )

        print("=" * 60)

        # ====================================================
        # 21. TAG EXACT CANDIDATE VERSION
        # ====================================================

        print("\n" + "=" * 60)
        print("TAGGING CANDIDATE MODEL VERSION")
        print("=" * 60)

        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=candidate_version_number,
            key="lifecycle",
            value="candidate",
        )

        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=candidate_version_number,
            key="model_type",
            value=str(best_model_name),
        )

        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=candidate_version_number,
            key="training_run_id",
            value=run_id,
        )

        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=candidate_version_number,
            key="pipeline",
            value="continuous_deployment_pipeline",
        )

        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=candidate_version_number,
            key="candidate_status",
            value="awaiting_quality_gate",
        )

        print(
            "Candidate tags applied successfully."
        )

        print("=" * 60)

        # ====================================================
        # 22. VERIFY EXACT REGISTERED VERSION
        # ====================================================

        print("\n" + "=" * 60)
        print("VERIFYING REGISTERED CANDIDATE")
        print("=" * 60)

        registered_version = (
            client.get_model_version(
                name=REGISTERED_MODEL_NAME,
                version=candidate_version_number,
            )
        )

        if (
            registered_version.run_id
            != run_id
        ):
            raise RuntimeError(
                "Registered model version does "
                "not belong to the current "
                "training run.\n"
                f"Expected Run ID: {run_id}\n"
                f"Actual Run ID: "
                f"{registered_version.run_id}"
            )

        lifecycle = (
            registered_version.tags.get(
                "lifecycle"
            )
        )

        if lifecycle != "candidate":
            raise RuntimeError(
                "Registered model version was "
                "not tagged as candidate.\n"
                f"Actual lifecycle: {lifecycle}"
            )

        print(
            f"Verified Version : "
            f"{registered_version.version}"
        )

        print(
            f"Verified Run ID   : "
            f"{registered_version.run_id}"
        )

        print(
            f"Lifecycle         : "
            f"{lifecycle}"
        )

        print("=" * 60)

        # ====================================================
        # 23. VERIFY CHAMPION WAS NOT MODIFIED
        # ====================================================

        print("\n" + "=" * 60)
        print("VERIFYING CHAMPION ALIAS")
        print("=" * 60)

        champion_version = None

        try:

            existing_champion = (
                client.get_model_version_by_alias(
                    REGISTERED_MODEL_NAME,
                    MODEL_ALIAS,
                )
            )

            champion_version = str(
                existing_champion.version
            )

            print(
                f"Existing Champion Version: "
                f"{champion_version}"
            )

            if (
                champion_version
                == candidate_version_number
            ):
                print(
                    "Warning: candidate is already "
                    "the champion."
                )

        except Exception:

            print(
                "No existing champion alias found."
            )

        print(
            "\nTraining step does NOT modify "
            "the champion alias."
        )

        print("=" * 60)

        # ====================================================
        # 24. FINAL TRAINING INFORMATION
        # ====================================================

        print("\n" + "=" * 60)
        print("FINAL MODEL TRAINED SUCCESSFULLY")
        print("=" * 60)

        print(
            f"Model                  : "
            f"{best_model_name}"
        )

        print(
            f"Registered Model       : "
            f"{REGISTERED_MODEL_NAME}"
        )

        print(
            f"Candidate Version      : "
            f"{candidate_version_number}"
        )

        print(
            "Candidate Lifecycle    : candidate"
        )

        print(
            f"Champion Alias         : "
            f"{MODEL_ALIAS}"
        )

        if champion_version is not None:

            print(
                f"Current Champion       : "
                f"{champion_version}"
            )

        else:

            print(
                "Current Champion       : None"
            )

        print(
            f"Baseline CV R²         : "
            f"{baseline_cv_r2:.6f}"
        )

        print(
            f"Final CV R²            : "
            f"{tuned_cv_r2:.6f}"
        )

        print(
            f"Final CV RMSE          : "
            f"{tuned_cv_rmse:.6f}"
        )

        print(
            f"Final CV MAE           : "
            f"{tuned_cv_mae:.6f}"
        )

        print(
            f"Run ID                 : "
            f"{run_id}"
        )

        print(
            f"Tracking URI           : "
            f"{mlflow.get_tracking_uri()}"
        )

        print("=" * 60)

        # ====================================================
        # 25. RETURN MODEL + EXACT CANDIDATE VERSION
        # ====================================================

        return (
            trained_model,
            candidate_version_number,
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as exc:

        logging.exception(
            "Model training failed."
        )

        print("\n" + "=" * 60)
        print("ERROR DURING MODEL TRAINING")
        print("=" * 60)

        print(
            f"Error: {str(exc)}"
        )

        print("=" * 60)

        raise