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

CV_SPLITS = 5
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
            n_splits=CV_SPLITS,
            random_state=RANDOM_STATE,
        )

        results = benchmark.compare_models(
            candidate_models,
            x_train,
            y_train,
        )

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

        best_cv_r2 = float(
            results.iloc[0]["cv_r2"]
        )

        print("\n" + "=" * 60)
        print(
            f"BEST MODEL SELECTED: "
            f"{best_model_name}"
        )
        print(
            f"BEST CV R²: "
            f"{best_cv_r2:.4f}"
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
            "cv_splits",
            CV_SPLITS,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )


        # ====================================================
        # 5. CREATE SELECTED MODEL
        #
        # IMPORTANT:
        # No MLflow autologging here.
        #
        # Optuna will train multiple configurations.
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
        #
        # IMPORTANT:
        # x_test and y_test are NOT used here.
        #
        # Optuna uses K-Fold CV on training data.
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
                n_splits=CV_SPLITS,
                random_state=RANDOM_STATE,
            )

            best_params = tuner.optimize(
                n_trials=OPTUNA_TRIALS,
            )

            print("\nBest parameters:")
            print(best_params)

            print(
                "\nBest parameters will be used "
                "for the final model."
            )

            mlflow.log_param(
                "optuna_enabled",
                True,
            )

            mlflow.log_param(
                "optuna_trials",
                OPTUNA_TRIALS,
            )

        else:

            best_params = {}

            mlflow.log_param(
                "optuna_enabled",
                False,
            )


        # ====================================================
        # 7. ENABLE MLFLOW AUTOLOGGING
        #
        # ONLY FOR FINAL MODEL
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
        # ====================================================

        if ENABLE_FINE_TUNING:

            for param_name, param_value in best_params.items():

                mlflow.log_param(
                    f"tuned_{param_name}",
                    param_value,
                )


        # ====================================================
        # 9. TRAIN FINAL MODEL
        #
        # Uses ALL training data.
        #
        # Test data remains untouched.
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
            f"Model: {best_model_name}"
        )

        print(
            f"Benchmark CV R²: "
            f"{best_cv_r2:.4f}"
        )

        if ENABLE_FINE_TUNING:

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