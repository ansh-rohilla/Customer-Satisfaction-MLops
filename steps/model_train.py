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
                    "cv_rmse_std",
                    "cv_mae",
                    "cv_mae_std",
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
        # 5. CREATE SELECTED MODEL
        #
        # IMPORTANT:
        # DO NOT ENABLE MLFLOW AUTOLOGGING HERE.
        #
        # Optuna trains many models during optimization.
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


            # =================================================
            # EXTRACT TUNING RESULTS
            # =================================================

            best_params = tuning_results[
                "best_params"
            ]

            tuned_cv_r2 = float(
                tuning_results[
                    "best_cv_r2"
                ]
            )

            tuned_cv_r2_std = float(
                tuning_results[
                    "cv_r2_std"
                ]
            )

            tuned_cv_rmse = float(
                tuning_results[
                    "cv_rmse"
                ]
            )

            tuned_cv_rmse_std = float(
                tuning_results[
                    "cv_rmse_std"
                ]
            )

            tuned_cv_mae = float(
                tuning_results[
                    "cv_mae"
                ]
            )

            tuned_cv_mae_std = float(
                tuning_results[
                    "cv_mae_std"
                ]
            )

            # ------------------------------------------------
            # NEW:
            # Trial history replaces the old Optuna study.
            # ------------------------------------------------

            trial_history = tuning_results[
                "trial_history"
            ]

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
            # 7. MLFLOW OPTUNA METADATA
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
                tuning_results[
                    "sampler"
                ],
            )

            mlflow.log_param(
                "optuna_direction",
                tuning_results[
                    "direction"
                ],
            )

            mlflow.log_param(
                "optuna_best_trial_number",
                best_trial_number,
            )


            # =================================================
            # 8. MLFLOW CV METRICS
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
            # 9. MLFLOW IMPROVEMENT METRICS
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
            # 10. LOG BEST PARAMETERS
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
            # 11. LOG INDIVIDUAL OPTUNA TRIALS
            #
            # trial_history comes from model_dev.py.
            # No raw Optuna Study is required.
            # =================================================

            print("\n" + "=" * 60)
            print("OPTUNA TRIAL SUMMARY")
            print("=" * 60)

            for trial in trial_history:

                trial_number = int(
                    trial["trial_number"]
                )

                trial_state = trial[
                    "state"
                ]

                trial_value = trial[
                    "value"
                ]

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


                # ---------------------------------------------
                # Trial R²
                # ---------------------------------------------

                mlflow.log_metric(
                    f"optuna_trial_{trial_number}_r2",
                    trial_r2,
                )


                # ---------------------------------------------
                # Trial CV R² Std
                # ---------------------------------------------

                if (
                    trial["cv_r2_std"]
                    is not None
                ):

                    mlflow.log_metric(
                        f"optuna_trial_{trial_number}_r2_std",
                        float(
                            trial[
                                "cv_r2_std"
                            ]
                        ),
                    )


                # ---------------------------------------------
                # Trial RMSE
                # ---------------------------------------------

                if (
                    trial["cv_rmse"]
                    is not None
                ):

                    mlflow.log_metric(
                        f"optuna_trial_{trial_number}_rmse",
                        float(
                            trial[
                                "cv_rmse"
                            ]
                        ),
                    )


                # ---------------------------------------------
                # Trial RMSE Std
                # ---------------------------------------------

                if (
                    trial["cv_rmse_std"]
                    is not None
                ):

                    mlflow.log_metric(
                        f"optuna_trial_{trial_number}_rmse_std",
                        float(
                            trial[
                                "cv_rmse_std"
                            ]
                        ),
                    )


                # ---------------------------------------------
                # Trial MAE
                # ---------------------------------------------

                if (
                    trial["cv_mae"]
                    is not None
                ):

                    mlflow.log_metric(
                        f"optuna_trial_{trial_number}_mae",
                        float(
                            trial[
                                "cv_mae"
                            ]
                        ),
                    )


                # ---------------------------------------------
                # Trial MAE Std
                # ---------------------------------------------

                if (
                    trial["cv_mae_std"]
                    is not None
                ):

                    mlflow.log_metric(
                        f"optuna_trial_{trial_number}_mae_std",
                        float(
                            trial[
                                "cv_mae_std"
                            ]
                        ),
                    )


                # ---------------------------------------------
                # Trial Duration
                # ---------------------------------------------

                if (
                    trial[
                        "duration_seconds"
                    ]
                    is not None
                ):

                    mlflow.log_metric(
                        f"optuna_trial_{trial_number}_duration_seconds",
                        float(
                            trial[
                                "duration_seconds"
                            ]
                        ),
                    )

            print("=" * 60)


            # =================================================
            # 12. LOG BEST TRIAL INFORMATION
            # =================================================

            mlflow.log_param(
                "optuna_best_trial_number",
                best_trial_number,
            )

            mlflow.log_metric(
                "optuna_best_trial_r2",
                optuna_best_value,
            )


        # ====================================================
        # 13. FINE TUNING DISABLED
        # ====================================================

        else:

            best_params = {}

            tuned_cv_r2 = (
                baseline_cv_r2
            )

            cv_r2_improvement = 0.0

            cv_r2_improvement_pct = 0.0

            mlflow.log_param(
                "optuna_enabled",
                False,
            )


        # ====================================================
        # 14. ENABLE MLFLOW AUTOLOGGING
        #
        # ONLY FOR FINAL MODEL.
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
        # 15. TRAIN FINAL MODEL
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
        # 16. FINAL MODEL INFORMATION
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
        # 17. RETURN FINAL MODEL
        # ====================================================

        return trained_model


    except Exception as e:

        logging.error(
            f"Model training failed: {e}"
        )

        raise