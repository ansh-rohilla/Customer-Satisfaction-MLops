import logging

import mlflow
import pandas as pd

from model.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
    ModelBenchmark,
    get_candidate_models,
)

from sklearn.base import RegressorMixin

from zenml import step
from zenml.client import Client


# ============================================================
# CONFIGURATION
# ============================================================

ENABLE_FINE_TUNING = True
OPTUNA_TRIALS = 10

CV_FOLDS = 5
RANDOM_STATE = 42


# ============================================================
# MLFLOW EXPERIMENT TRACKER
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
) -> RegressorMixin:

    try:

        # ====================================================
        # 1. MODEL BENCHMARKING
        # ====================================================

        print("\n" + "=" * 60)
        print("STARTING MODEL BENCHMARKING")
        print("=" * 60)

        candidate_models = get_candidate_models()

        benchmark = ModelBenchmark(
            n_splits=CV_FOLDS,
            random_state=RANDOM_STATE,
        )

        results = benchmark.compare_models(
            candidate_models,
            x_train,
            y_train,
        )

        # Make sure best model is first
        results = (
            results
            .sort_values(
                by="cv_r2",
                ascending=False,
            )
            .reset_index(drop=True)
        )


        # ====================================================
        # 2. DISPLAY MODEL COMPARISON
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
                    "cv_mae",
                ]
            ].to_string(index=False)
        )


        # ====================================================
        # 3. SELECT BEST MODEL
        # ====================================================

        best_model_name = results.iloc[0]["model"]

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
        # 4. LOG BENCHMARK RESULTS TO MLFLOW
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
                f"{model_name}_cv_mae",
                float(row["cv_mae"]),
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
        # 5. CREATE SELECTED MODEL
        #
        # IMPORTANT:
        # DO NOT ENABLE MLFLOW AUTOLOGGING HERE.
        #
        # Optuna will train many models.
        # Enabling autologging here can cause MLflow
        # parameter conflicts between trials.
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
                f"Unsupported model: {best_model_name}"
            )


        # ====================================================
        # 6. OPTUNA HYPERPARAMETER TUNING
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

            # ------------------------------------------------
            # Extract tuning results
            # ------------------------------------------------

            best_params = tuning_results[
                "best_params"
            ]

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


            # ------------------------------------------------
            # Calculate improvement
            # ------------------------------------------------

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
            # DISPLAY OPTUNA RESULTS
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
                f"Best Parameters: "
                f"{best_params}"
            )

            print("=" * 60)


            # =================================================
            # LOG OPTUNA INFORMATION TO MLFLOW
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


            # ------------------------------------------------
            # CV METRICS
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


            # ------------------------------------------------
            # Improvement metrics
            # ------------------------------------------------

            mlflow.log_metric(
                "cv_r2_improvement",
                cv_r2_improvement,
            )

            mlflow.log_metric(
                "cv_r2_improvement_pct",
                cv_r2_improvement_pct,
            )

        else:

            # =================================================
            # FINE TUNING DISABLED
            # =================================================

            best_params = {}

            tuned_cv_r2 = baseline_cv_r2

            mlflow.log_param(
                "optuna_enabled",
                False,
            )


        # ====================================================
        # 7. ENABLE MLFLOW AUTOLOGGING
        #
        # ONLY FOR THE FINAL MODEL.
        # ====================================================

        print("\n" + "=" * 60)
        print("ENABLING MLFLOW AUTOLOGGING")
        print("=" * 60)

        if best_model_name == "lightgbm":

            mlflow.lightgbm.autolog()

        elif best_model_name == "xgboost":

            mlflow.xgboost.autolog()

        else:

            mlflow.sklearn.autolog()


        # ====================================================
        # 8. LOG FINAL OPTUNA PARAMETERS
        #
        # Use unique parameter names so they don't conflict
        # with MLflow's final-model autologging.
        # ====================================================

        if ENABLE_FINE_TUNING:

            for (
                param_name,
                param_value
            ) in best_params.items():

                mlflow.log_param(
                    f"tuned_{param_name}",
                    param_value,
                )


        # ====================================================
        # 9. TRAIN FINAL MODEL
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
        # 10. FINAL MODEL INFORMATION
        # ====================================================

        print("\n" + "=" * 60)
        print("FINAL MODEL TRAINED SUCCESSFULLY")
        print("=" * 60)

        print(
            f"Model: "
            f"{best_model_name}"
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
        # 11. RETURN FINAL MODEL
        # ====================================================

        return trained_model


    except Exception as e:

        logging.error(
            f"Model training failed: {e}"
        )

        raise