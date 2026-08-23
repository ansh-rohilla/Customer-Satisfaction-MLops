import logging
import os
from pathlib import Path
from typing import Any, Tuple

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient

from model.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
    ModelBenchmark,
    get_candidate_models,
)

from zenml import step
from zenml.client import Client


# ============================================================
# ENVIRONMENT / MLFLOW CONFIGURATION
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# MLFLOW TRACKING URI
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}",
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


print("\n" + "=" * 60)
print("MLFLOW CONFIGURATION")
print("=" * 60)

print(
    f"Tracking URI: "
    f"{mlflow.get_tracking_uri()}"
)

print("=" * 60)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

ENABLE_FINE_TUNING = True

OPTUNA_TRIALS = 10

CV_FOLDS = 5

RANDOM_STATE = 42


# ============================================================
# MLFLOW MODEL REGISTRY
# ============================================================

REGISTERED_MODEL_NAME = (
    "Customer-Satisfaction-Model"
)

MODEL_ALIAS = "champion"


# ============================================================
# ZENML MLFLOW EXPERIMENT TRACKER
# ============================================================

experiment_tracker = (
    Client()
    .active_stack
    .experiment_tracker
)


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

    try:

        # ====================================================
        # 1. VERIFY MLFLOW CONNECTION
        # ====================================================

        print("\n" + "=" * 60)
        print("VERIFYING MLFLOW")
        print("=" * 60)

        print(
            f"Tracking URI: "
            f"{mlflow.get_tracking_uri()}"
        )

        client = MlflowClient(
            tracking_uri=(
                mlflow.get_tracking_uri()
            )
        )

        print(
            "MLflow client initialized successfully."
        )

        print("=" * 60)


        # ====================================================
        # 2. MODEL BENCHMARKING
        # ====================================================

        print("\n" + "=" * 60)
        print("STARTING MODEL BENCHMARKING")
        print("=" * 60)

        candidate_models = (
            get_candidate_models()
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


        # ====================================================
        # 3. SORT MODELS BY CV R²
        # ====================================================

        results = (
            results
            .sort_values(
                by="cv_r2",
                ascending=False,
            )
            .reset_index(drop=True)
        )


        # ====================================================
        # 4. DISPLAY MODEL COMPARISON
        # ====================================================

        print("\n" + "=" * 60)
        print("MODEL COMPARISON RESULTS")
        print("=" * 60)

        print(
            results[
                [
                    "model",
                    "cv_r2",
                    "cv_r2_std",
                    "cv_rmse",
                    "cv_rmse_std",
                    "cv_mae",
                    "cv_mae_std",
                ]
            ].to_string(index=False)
        )


        # ====================================================
        # 5. SELECT BEST MODEL
        # ====================================================

        best_model_name = (
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
            f"{baseline_cv_r2:.4f}"
        )

        print("=" * 60)


        # ====================================================
        # 6. LOG BENCHMARK RESULTS
        # ====================================================

        for _, row in results.iterrows():

            model_name = row["model"]

            mlflow.log_metric(
                f"{model_name}_cv_r2",
                float(row["cv_r2"]),
            )

            mlflow.log_metric(
                f"{model_name}_cv_r2_std",
                float(row["cv_r2_std"]),
            )

            mlflow.log_metric(
                f"{model_name}_cv_rmse",
                float(row["cv_rmse"]),
            )

            mlflow.log_metric(
                f"{model_name}_cv_rmse_std",
                float(row["cv_rmse_std"]),
            )

            mlflow.log_metric(
                f"{model_name}_cv_mae",
                float(row["cv_mae"]),
            )

            mlflow.log_metric(
                f"{model_name}_cv_mae_std",
                float(row["cv_mae_std"]),
            )


        mlflow.log_param(
            "selected_model",
            best_model_name,
        )

        mlflow.log_param(
            "benchmark_cv_folds",
            CV_FOLDS,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )


        # ====================================================
        # 7. CREATE SELECTED MODEL
        # ====================================================

        if best_model_name == "lightgbm":

            model = LightGBMModel()

        elif best_model_name == "xgboost":

            model = XGBoostModel()

        elif best_model_name == "randomforest":

            model = RandomForestModel()

        elif best_model_name == "linear_regression":

            model = LinearRegressionModel()

        else:

            raise ValueError(
                f"Unsupported model: "
                f"{best_model_name}"
            )


        # ====================================================
        # 8. OPTUNA HYPERPARAMETER TUNING
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


            # =================================================
            # EXTRACT RESULTS
            # =================================================

            best_params = (
                tuning_results["best_params"]
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
                tuning_results["trial_history"]
            )

            best_trial_number = int(
                tuning_results[
                    "best_trial_number"
                ]
            )

            optuna_best_value = float(
                tuning_results[
                    "optuna_best_value"
                ]
            )


            # =================================================
            # CALCULATE IMPROVEMENT
            # =================================================

            cv_r2_improvement = (
                tuned_cv_r2
                - baseline_cv_r2
            )

            if baseline_cv_r2 != 0:

                cv_r2_improvement_pct = (
                    cv_r2_improvement
                    / abs(baseline_cv_r2)
                ) * 100

            else:

                cv_r2_improvement_pct = 0.0


            # =================================================
            # OPTUNA REPORT
            # =================================================

            print("\n" + "=" * 60)
            print("OPTUNA RESULTS")
            print("=" * 60)

            print(
                f"Baseline CV R² : "
                f"{baseline_cv_r2:.4f}"
            )

            print(
                f"Tuned CV R²    : "
                f"{tuned_cv_r2:.4f}"
            )

            print(
                f"CV R² Std      : "
                f"{tuned_cv_r2_std:.4f}"
            )

            print(
                f"CV RMSE        : "
                f"{tuned_cv_rmse:.4f}"
            )

            print(
                f"CV RMSE Std    : "
                f"{tuned_cv_rmse_std:.4f}"
            )

            print(
                f"CV MAE         : "
                f"{tuned_cv_mae:.4f}"
            )

            print(
                f"CV MAE Std     : "
                f"{tuned_cv_mae_std:.4f}"
            )

            print(
                f"R² Improvement : "
                f"{cv_r2_improvement:.4f}"
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


            # =================================================
            # MLFLOW OPTUNA METADATA
            # =================================================

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
                "optuna_sampler",
                tuning_results["sampler"],
            )

            mlflow.log_param(
                "optuna_direction",
                tuning_results["direction"],
            )

            mlflow.log_param(
                "optuna_best_trial_number",
                best_trial_number,
            )


            # =================================================
            # CV METRICS
            # =================================================

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


            # =================================================
            # IMPROVEMENT METRICS
            # =================================================

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


            # =================================================
            # BEST PARAMETERS
            # =================================================

            for (
                param_name,
                param_value,
            ) in best_params.items():

                mlflow.log_param(
                    f"tuned_{param_name}",
                    param_value,
                )


            # =================================================
            # OPTUNA TRIAL HISTORY
            # =================================================

            print("\n" + "=" * 60)
            print("OPTUNA TRIAL SUMMARY")
            print("=" * 60)

            for trial in trial_history:

                trial_number = int(
                    trial["trial_number"]
                )

                trial_state = (
                    trial["state"]
                )

                trial_value = (
                    trial["value"]
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
                    f"{trial_r2:.4f} "
                    f"| Params: "
                    f"{trial['params']}"
                )

                mlflow.log_metric(
                    f"optuna_trial_{trial_number}_r2",
                    trial_r2,
                )

                if trial["cv_r2_std"] is not None:

                    mlflow.log_metric(
                        f"optuna_trial_{trial_number}_r2_std",
                        float(
                            trial["cv_r2_std"]
                        ),
                    )

                if trial["cv_rmse"] is not None:

                    mlflow.log_metric(
                        f"optuna_trial_{trial_number}_rmse",
                        float(
                            trial["cv_rmse"]
                        ),
                    )

                if trial["cv_rmse_std"] is not None:

                    mlflow.log_metric(
                        f"optuna_trial_{trial_number}_rmse_std",
                        float(
                            trial["cv_rmse_std"]
                        ),
                    )

                if trial["cv_mae"] is not None:

                    mlflow.log_metric(
                        f"optuna_trial_{trial_number}_mae",
                        float(
                            trial["cv_mae"]
                        ),
                    )

                if trial["cv_mae_std"] is not None:

                    mlflow.log_metric(
                        f"optuna_trial_{trial_number}_mae_std",
                        float(
                            trial["cv_mae_std"]
                        ),
                    )

                if trial["duration_seconds"] is not None:

                    mlflow.log_metric(
                        f"optuna_trial_{trial_number}_duration_seconds",
                        float(
                            trial[
                                "duration_seconds"
                            ]
                        ),
                    )

            print("=" * 60)


        # ====================================================
        # 9. FINE TUNING DISABLED
        # ====================================================

        else:

            best_params = {}

            tuned_cv_r2 = (
                baseline_cv_r2
            )

            tuned_cv_r2_std = float(
                results.iloc[0][
                    "cv_r2_std"
                ]
            )

            tuned_cv_rmse = float(
                results.iloc[0][
                    "cv_rmse"
                ]
            )

            tuned_cv_rmse_std = float(
                results.iloc[0][
                    "cv_rmse_std"
                ]
            )

            tuned_cv_mae = float(
                results.iloc[0][
                    "cv_mae"
                ]
            )

            tuned_cv_mae_std = float(
                results.iloc[0][
                    "cv_mae_std"
                ]
            )

            cv_r2_improvement = 0.0

            cv_r2_improvement_pct = 0.0

            mlflow.log_param(
                "optuna_enabled",
                False,
            )


        # ====================================================
        # 10. ENABLE MLFLOW AUTOLOGGING
        # ====================================================

        print("\n" + "=" * 60)
        print("ENABLING MLFLOW AUTOLOGGING")
        print("=" * 60)

        print(
            f"Registered Model: "
            f"{REGISTERED_MODEL_NAME}"
        )


        if best_model_name == "lightgbm":

            mlflow.lightgbm.autolog(
                registered_model_name=(
                    REGISTERED_MODEL_NAME
                )
            )

        elif best_model_name == "xgboost":

            mlflow.xgboost.autolog(
                registered_model_name=(
                    REGISTERED_MODEL_NAME
                )
            )

        else:

            mlflow.sklearn.autolog(
                registered_model_name=(
                    REGISTERED_MODEL_NAME
                )
            )


        # ====================================================
        # 11. TRAIN FINAL MODEL
        # ====================================================

        print("\n" + "=" * 60)
        print("TRAINING FINAL MODEL")
        print("=" * 60)


        if ENABLE_FINE_TUNING:

            trained_model = model.train(
                x_train,
                y_train,
                **best_params,
            )

        else:

            trained_model = model.train(
                x_train,
                y_train,
            )


        # ====================================================
        # 12. VERIFY ACTIVE MLFLOW RUN
        # ====================================================

        active_run = mlflow.active_run()

        if active_run is None:

            raise RuntimeError(
                "No active MLflow run found "
                "after final model training."
            )

        run_id = (
            active_run.info.run_id
        )


        print("\n" + "=" * 60)
        print("MLFLOW RUN INFORMATION")
        print("=" * 60)

        print(
            f"Run ID: {run_id}"
        )

        print(
            f"Tracking URI: "
            f"{mlflow.get_tracking_uri()}"
        )

        print("=" * 60)


        # ====================================================
        # 13. CREATE TRAINING REFERENCE PROFILE
        # ====================================================

        print("\n" + "=" * 60)
        print("CREATING TRAINING REFERENCE PROFILE")
        print("=" * 60)

        reference_profile = {}


        for column in x_train.columns:

            numeric_values = pd.to_numeric(
                x_train[column],
                errors="coerce",
            ).dropna()

            if numeric_values.empty:
                continue

            reference_profile[column] = (
                numeric_values
                .astype(float)
                .tolist()
            )


        # ====================================================
        # 14. LOG REFERENCE PROFILE
        # ====================================================

        mlflow.log_dict(
            reference_profile,
            "monitoring/reference_profile.json",
        )

        print(
            "Training reference profile logged:"
        )

        print(
            "monitoring/reference_profile.json"
        )

        print(
            f"Reference features: "
            f"{len(reference_profile)}"
        )


        for feature, values in (
            reference_profile.items()
        ):

            print(
                f"  {feature:<35}"
                f" samples={len(values)}"
            )

        print("=" * 60)


        # ====================================================
        # 15. LOG MODEL METADATA
        # ====================================================

        mlflow.set_tag(
            "model_name",
            best_model_name,
        )

        mlflow.set_tag(
            "model_registry_name",
            REGISTERED_MODEL_NAME,
        )

        # IMPORTANT:
        # Training creates a CANDIDATE.
        # Deployment pipeline decides whether
        # the candidate becomes CHAMPION.

        mlflow.set_tag(
            "model_lifecycle",
            "candidate",
        )

        mlflow.log_param(
            "registered_model_name",
            REGISTERED_MODEL_NAME,
        )

        mlflow.log_param(
            "model_alias",
            MODEL_ALIAS,
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
        # 16. FIND EXACT REGISTERED VERSION FOR THIS RUN
        # ====================================================

        print("\n" + "=" * 60)
        print("CHECKING MLFLOW MODEL REGISTRY")
        print("=" * 60)

        model_versions = (
            client.search_model_versions(
                filter_string=(
                    f"name='{REGISTERED_MODEL_NAME}'"
                )
            )
        )

        matching_versions = [
            version
            for version in model_versions
            if version.run_id == run_id
        ]

        if not matching_versions:
            raise RuntimeError(
                "The final model was not found "
                "in the MLflow Model Registry.\n"
                f"Registered model: "
                f"{REGISTERED_MODEL_NAME}\n"
                f"Run ID: {run_id}\n"
                f"Tracking URI: "
                f"{mlflow.get_tracking_uri()}"
            )

        candidate_version = max(
            matching_versions,
            key=lambda version: int(
                version.version
            ),
        )

        # IMPORTANT:
        # ZenML expects this output to be a string.
        candidate_version_number = str(
            candidate_version.version
        )

        print(
            f"Registered Model: "
            f"{REGISTERED_MODEL_NAME}"
        )

        print(
            f"Candidate Version: "
            f"{candidate_version_number}"
        )

        print(
            f"Training Run ID: "
            f"{run_id}"
        )

        print(
            "Lifecycle: candidate"
        )

        print("=" * 60)

        # ====================================================
        # 17. TAG REGISTERED MODEL VERSION
        # ====================================================

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

        # ====================================================
        # 18. FINAL MODEL INFORMATION
        # ====================================================

        print("\n" + "=" * 60)
        print("FINAL MODEL TRAINED SUCCESSFULLY")
        print("=" * 60)

        print(
            f"Model: "
            f"{best_model_name}"
        )

        print(
            f"MLflow Registered Model: "
            f"{REGISTERED_MODEL_NAME}"
        )

        print(
            f"MLflow Candidate Version: "
            f"{candidate_version_number}"
        )

        print(
            "MLflow Lifecycle: candidate"
        )

        print(
            "MLflow Alias: "
            "NOT ASSIGNED BY TRAINING"
        )

        print(
            f"Baseline CV R²: "
            f"{baseline_cv_r2:.4f}"
        )

        if ENABLE_FINE_TUNING:

            print(
                f"Tuned CV R²: "
                f"{tuned_cv_r2:.4f}"
            )

            print(
                f"CV R² Improvement: "
                f"{cv_r2_improvement:.4f}"
            )

            print(
                f"CV R² Improvement %: "
                f"{cv_r2_improvement_pct:.2f}%"
            )

            print(
                f"Optuna Trials: "
                f"{OPTUNA_TRIALS}"
            )

            print(
                f"Best Parameters: "
                f"{best_params}"
            )

        print("=" * 60)

        # ====================================================
        # 19. RETURN MODEL + CANDIDATE VERSION
        # ====================================================

        return (
            trained_model,
            candidate_version_number,
        )

    except Exception:

        logging.exception(
            "Model training failed."
        )

        raise