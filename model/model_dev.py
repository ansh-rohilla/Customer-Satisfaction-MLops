import optuna
import pandas as pd
import xgboost as xgb

from abc import ABC, abstractmethod

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

DEFAULT_CV_FOLDS = 5
DEFAULT_RANDOM_STATE = 42


# ============================================================
# CROSS-VALIDATION EVALUATION
# ============================================================

def evaluate_model_cv(
    model,
    x_train,
    y_train,
    n_splits=DEFAULT_CV_FOLDS,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Evaluate a regression model using K-Fold Cross-Validation.

    Metrics:
        - R²
        - RMSE
        - MAE

    Returns:
        Dictionary containing mean and standard deviation
        for all metrics.
    """

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    return {
        # R²
        "cv_r2_mean": float(
            scores["test_r2"].mean()
        ),
        "cv_r2_std": float(
            scores["test_r2"].std()
        ),

        # RMSE
        "cv_rmse_mean": float(
            -scores["test_rmse"].mean()
        ),
        "cv_rmse_std": float(
            scores["test_rmse"].std()
        ),

        # MAE
        "cv_mae_mean": float(
            -scores["test_mae"].mean()
        ),
        "cv_mae_std": float(
            scores["test_mae"].std()
        ),
    }


# ============================================================
# BASE MODEL
# ============================================================

class Model(ABC):

    @abstractmethod
    def train(
        self,
        x_train,
        y_train,
        **kwargs,
    ):
        pass

    @abstractmethod
    def optimize_cv(
        self,
        trial,
        x_train,
        y_train,
        n_splits=DEFAULT_CV_FOLDS,
        random_state=DEFAULT_RANDOM_STATE,
    ):
        pass


# ============================================================
# RANDOM FOREST
# ============================================================

class RandomForestModel(Model):

    def train(
        self,
        x_train,
        y_train,
        **kwargs,
    ):

        reg = RandomForestRegressor(
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
            **kwargs,
        )

        reg.fit(
            x_train,
            y_train,
        )

        return reg

    def optimize_cv(
        self,
        trial,
        x_train,
        y_train,
        n_splits=DEFAULT_CV_FOLDS,
        random_state=DEFAULT_RANDOM_STATE,
    ):

        n_estimators = trial.suggest_int(
            "n_estimators",
            100,
            500,
        )

        max_depth = trial.suggest_int(
            "max_depth",
            5,
            30,
        )

        min_samples_split = trial.suggest_int(
            "min_samples_split",
            2,
            20,
        )

        model = RandomForestRegressor(
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )

        metrics = evaluate_model_cv(
            model,
            x_train,
            y_train,
            n_splits=n_splits,
            random_state=random_state,
        )

        self._store_trial_metrics(
            trial,
            metrics,
        )

        return metrics["cv_r2_mean"]

    @staticmethod
    def _store_trial_metrics(
        trial,
        metrics,
    ):

        trial.set_user_attr(
            "cv_r2_std",
            float(metrics["cv_r2_std"]),
        )

        trial.set_user_attr(
            "cv_rmse",
            float(metrics["cv_rmse_mean"]),
        )

        trial.set_user_attr(
            "cv_rmse_std",
            float(metrics["cv_rmse_std"]),
        )

        trial.set_user_attr(
            "cv_mae",
            float(metrics["cv_mae_mean"]),
        )

        trial.set_user_attr(
            "cv_mae_std",
            float(metrics["cv_mae_std"]),
        )


# ============================================================
# LIGHTGBM
# ============================================================

class LightGBMModel(Model):

    def train(
        self,
        x_train,
        y_train,
        **kwargs,
    ):

        reg = LGBMRegressor(
            random_state=DEFAULT_RANDOM_STATE,
            verbosity=-1,
            **kwargs,
        )

        reg.fit(
            x_train,
            y_train,
        )

        return reg

    def optimize_cv(
        self,
        trial,
        x_train,
        y_train,
        n_splits=DEFAULT_CV_FOLDS,
        random_state=DEFAULT_RANDOM_STATE,
    ):

        n_estimators = trial.suggest_int(
            "n_estimators",
            100,
            1000,
        )

        max_depth = trial.suggest_int(
            "max_depth",
            3,
            15,
        )

        learning_rate = trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True,
        )

        num_leaves = trial.suggest_int(
            "num_leaves",
            20,
            150,
        )

        model = LGBMRegressor(
            random_state=DEFAULT_RANDOM_STATE,
            verbosity=-1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
        )

        metrics = evaluate_model_cv(
            model,
            x_train,
            y_train,
            n_splits=n_splits,
            random_state=random_state,
        )

        self._store_trial_metrics(
            trial,
            metrics,
        )

        return metrics["cv_r2_mean"]

    @staticmethod
    def _store_trial_metrics(
        trial,
        metrics,
    ):

        trial.set_user_attr(
            "cv_r2_std",
            float(metrics["cv_r2_std"]),
        )

        trial.set_user_attr(
            "cv_rmse",
            float(metrics["cv_rmse_mean"]),
        )

        trial.set_user_attr(
            "cv_rmse_std",
            float(metrics["cv_rmse_std"]),
        )

        trial.set_user_attr(
            "cv_mae",
            float(metrics["cv_mae_mean"]),
        )

        trial.set_user_attr(
            "cv_mae_std",
            float(metrics["cv_mae_std"]),
        )


# ============================================================
# XGBOOST
# ============================================================

class XGBoostModel(Model):

    def train(
        self,
        x_train,
        y_train,
        **kwargs,
    ):

        reg = xgb.XGBRegressor(
            random_state=DEFAULT_RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=-1,
            **kwargs,
        )

        reg.fit(
            x_train,
            y_train,
        )

        return reg

    def optimize_cv(
        self,
        trial,
        x_train,
        y_train,
        n_splits=DEFAULT_CV_FOLDS,
        random_state=DEFAULT_RANDOM_STATE,
    ):

        n_estimators = trial.suggest_int(
            "n_estimators",
            100,
            1000,
        )

        max_depth = trial.suggest_int(
            "max_depth",
            3,
            12,
        )

        learning_rate = trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True,
        )

        subsample = trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        )

        colsample_bytree = trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0,
        )

        model = xgb.XGBRegressor(
            random_state=DEFAULT_RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=-1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        )

        metrics = evaluate_model_cv(
            model,
            x_train,
            y_train,
            n_splits=n_splits,
            random_state=random_state,
        )

        self._store_trial_metrics(
            trial,
            metrics,
        )

        return metrics["cv_r2_mean"]

    @staticmethod
    def _store_trial_metrics(
        trial,
        metrics,
    ):

        trial.set_user_attr(
            "cv_r2_std",
            float(metrics["cv_r2_std"]),
        )

        trial.set_user_attr(
            "cv_rmse",
            float(metrics["cv_rmse_mean"]),
        )

        trial.set_user_attr(
            "cv_rmse_std",
            float(metrics["cv_rmse_std"]),
        )

        trial.set_user_attr(
            "cv_mae",
            float(metrics["cv_mae_mean"]),
        )

        trial.set_user_attr(
            "cv_mae_std",
            float(metrics["cv_mae_std"]),
        )


# ============================================================
# LINEAR REGRESSION
# ============================================================

class LinearRegressionModel(Model):

    def train(
        self,
        x_train,
        y_train,
        **kwargs,
    ):

        reg = LinearRegression(
            **kwargs,
        )

        reg.fit(
            x_train,
            y_train,
        )

        return reg

    def optimize_cv(
        self,
        trial,
        x_train,
        y_train,
        n_splits=DEFAULT_CV_FOLDS,
        random_state=DEFAULT_RANDOM_STATE,
    ):

        model = LinearRegression()

        metrics = evaluate_model_cv(
            model,
            x_train,
            y_train,
            n_splits=n_splits,
            random_state=random_state,
        )

        self._store_trial_metrics(
            trial,
            metrics,
        )

        return metrics["cv_r2_mean"]

    @staticmethod
    def _store_trial_metrics(
        trial,
        metrics,
    ):

        trial.set_user_attr(
            "cv_r2_std",
            float(metrics["cv_r2_std"]),
        )

        trial.set_user_attr(
            "cv_rmse",
            float(metrics["cv_rmse_mean"]),
        )

        trial.set_user_attr(
            "cv_rmse_std",
            float(metrics["cv_rmse_std"]),
        )

        trial.set_user_attr(
            "cv_mae",
            float(metrics["cv_mae_mean"]),
        )

        trial.set_user_attr(
            "cv_mae_std",
            float(metrics["cv_mae_std"]),
        )


# ============================================================
# OPTUNA HYPERPARAMETER TUNER
# ============================================================

class HyperparameterTuner:

    """
    Performs Optuna hyperparameter optimization
    using K-Fold Cross-Validation.

    The test set is NEVER used here.

    Returns:
        Complete serializable Optuna reporting information.
    """

    def __init__(
        self,
        model,
        x_train,
        y_train,
        n_splits=DEFAULT_CV_FOLDS,
        random_state=DEFAULT_RANDOM_STATE,
    ):

        self.model = model
        self.x_train = x_train
        self.y_train = y_train

        self.n_splits = n_splits
        self.random_state = random_state

    def optimize(
        self,
        n_trials=30,
    ):

        # ====================================================
        # 1. CREATE OPTUNA STUDY
        # ====================================================

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        # ====================================================
        # 2. OPTUNA CALLBACK
        # ====================================================

        def print_trial_result(
            study,
            trial,
        ):

            if trial.state != optuna.trial.TrialState.COMPLETE:
                return

            print(
                f"Trial {trial.number + 1}/{n_trials} "
                f"| CV R²: {trial.value:.4f} "
                f"| Best R²: "
                f"{study.best_value:.4f}"
            )

        # ====================================================
        # 3. RUN OPTUNA
        # ====================================================

        study.optimize(
            lambda trial: self.model.optimize_cv(
                trial,
                self.x_train,
                self.y_train,
                n_splits=self.n_splits,
                random_state=self.random_state,
            ),
            n_trials=n_trials,
            callbacks=[
                print_trial_result,
            ],
        )

        # ====================================================
        # 4. BEST TRIAL
        # ====================================================

        best_trial = study.best_trial

        best_params = dict(
            best_trial.params
        )

        # ====================================================
        # 5. CREATE BEST MODEL
        # ====================================================

        best_model = self.model.train(
            self.x_train,
            self.y_train,
            **best_params,
        )

        # ====================================================
        # 6. FINAL CV EVALUATION
        # ====================================================

        cv_metrics = evaluate_model_cv(
            best_model,
            self.x_train,
            self.y_train,
            n_splits=self.n_splits,
            random_state=self.random_state,
        )

        # ====================================================
        # 7. BUILD TRIAL HISTORY
        # ====================================================

        trial_history = []

        for trial in study.trials:

            trial_record = {
                "trial_number": int(
                    trial.number
                ),

                "state": trial.state.name,

                "value": (
                    float(trial.value)
                    if trial.value is not None
                    else None
                ),

                "params": dict(
                    trial.params
                ),

                "cv_r2_std": (
                    float(
                        trial.user_attrs[
                            "cv_r2_std"
                        ]
                    )
                    if "cv_r2_std"
                    in trial.user_attrs
                    else None
                ),

                "cv_rmse": (
                    float(
                        trial.user_attrs[
                            "cv_rmse"
                        ]
                    )
                    if "cv_rmse"
                    in trial.user_attrs
                    else None
                ),

                "cv_rmse_std": (
                    float(
                        trial.user_attrs[
                            "cv_rmse_std"
                        ]
                    )
                    if "cv_rmse_std"
                    in trial.user_attrs
                    else None
                ),

                "cv_mae": (
                    float(
                        trial.user_attrs[
                            "cv_mae"
                        ]
                    )
                    if "cv_mae"
                    in trial.user_attrs
                    else None
                ),

                "cv_mae_std": (
                    float(
                        trial.user_attrs[
                            "cv_mae_std"
                        ]
                    )
                    if "cv_mae_std"
                    in trial.user_attrs
                    else None
                ),

                "duration_seconds": (
                    float(
                        trial.duration.total_seconds()
                    )
                    if trial.duration is not None
                    else None
                ),
            }

            trial_history.append(
                trial_record
            )

        # ====================================================
        # 8. FINAL OPTUNA RESULTS
        # ====================================================

        print("\n" + "=" * 60)
        print("OPTUNA FINAL CV RESULTS")
        print("=" * 60)

        print(
            f"Best Trial       : "
            f"{best_trial.number + 1}"
        )

        print(
            f"Best CV R²       : "
            f"{cv_metrics['cv_r2_mean']:.4f}"
        )

        print(
            f"CV R² Std        : "
            f"{cv_metrics['cv_r2_std']:.4f}"
        )

        print(
            f"CV RMSE          : "
            f"{cv_metrics['cv_rmse_mean']:.4f}"
        )

        print(
            f"CV RMSE Std      : "
            f"{cv_metrics['cv_rmse_std']:.4f}"
        )

        print(
            f"CV MAE           : "
            f"{cv_metrics['cv_mae_mean']:.4f}"
        )

        print(
            f"CV MAE Std       : "
            f"{cv_metrics['cv_mae_std']:.4f}"
        )

        print(
            f"Best Parameters  : "
            f"{best_params}"
        )

        print("=" * 60)

        # ====================================================
        # 9. RETURN COMPLETE SERIALIZABLE REPORT
        # ====================================================

        return {
            # Best trial
            "best_trial_number": int(
                best_trial.number
            ),

            "best_params": best_params,

            "optuna_best_value": float(
                best_trial.value
            ),

            # Final CV metrics
            "best_cv_r2": float(
                cv_metrics["cv_r2_mean"]
            ),

            "cv_r2_std": float(
                cv_metrics["cv_r2_std"]
            ),

            "cv_rmse": float(
                cv_metrics["cv_rmse_mean"]
            ),

            "cv_rmse_std": float(
                cv_metrics["cv_rmse_std"]
            ),

            "cv_mae": float(
                cv_metrics["cv_mae_mean"]
            ),

            "cv_mae_std": float(
                cv_metrics["cv_mae_std"]
            ),

            # Optuna configuration
            "n_trials": int(
                n_trials
            ),

            "n_splits": int(
                self.n_splits
            ),

            "random_state": int(
                self.random_state
            ),

            "direction": "maximize",

            "sampler": "TPESampler",

            # Complete trial history
            "trial_history": trial_history,
        }


# ============================================================
# MODEL BENCHMARK
# ============================================================

class ModelBenchmark:

    def __init__(
        self,
        n_splits=DEFAULT_CV_FOLDS,
        random_state=DEFAULT_RANDOM_STATE,
    ):

        self.n_splits = n_splits
        self.random_state = random_state

    def compare_models(
        self,
        models,
        x_train,
        y_train,
    ):

        results = []

        for model_name, model in models.items():

            print(
                f"\nBenchmarking {model_name}..."
            )

            metrics = evaluate_model_cv(
                model,
                x_train,
                y_train,
                n_splits=self.n_splits,
                random_state=self.random_state,
            )

            mean_r2 = metrics[
                "cv_r2_mean"
            ]

            std_r2 = metrics[
                "cv_r2_std"
            ]

            mean_rmse = metrics[
                "cv_rmse_mean"
            ]

            std_rmse = metrics[
                "cv_rmse_std"
            ]

            mean_mae = metrics[
                "cv_mae_mean"
            ]

            std_mae = metrics[
                "cv_mae_std"
            ]

            print(
                f"{model_name} → "
                f"R²={mean_r2:.4f} "
                f"± {std_r2:.4f}, "
                f"RMSE={mean_rmse:.4f} "
                f"± {std_rmse:.4f}, "
                f"MAE={mean_mae:.4f} "
                f"± {std_mae:.4f}"
            )

            results.append(
                {
                    "model": model_name,

                    "cv_r2": mean_r2,
                    "cv_r2_std": std_r2,

                    "cv_rmse": mean_rmse,
                    "cv_rmse_std": std_rmse,

                    "cv_mae": mean_mae,
                    "cv_mae_std": std_mae,
                }
            )

        # ====================================================
        # SORT BY R²
        # ====================================================

        results_df = pd.DataFrame(
            results
        )

        results_df = (
            results_df
            .sort_values(
                by="cv_r2",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        return results_df


# ============================================================
# CANDIDATE MODELS
# ============================================================

def get_candidate_models():

    return {

        "lightgbm": LGBMRegressor(
            random_state=DEFAULT_RANDOM_STATE,
            verbosity=-1,
        ),

        "xgboost": xgb.XGBRegressor(
            random_state=DEFAULT_RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=-1,
        ),

        "randomforest": RandomForestRegressor(
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
        ),

        "linear_regression": LinearRegression(),
    }