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

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if x_train is None or y_train is None:
        raise ValueError(
            "x_train and y_train cannot be None."
        )

    if len(x_train) == 0:
        raise ValueError(
            "x_train is empty."
        )

    if len(x_train) != len(y_train):
        raise ValueError(
            "x_train and y_train must contain "
            "the same number of samples."
        )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2."
        )

    if len(x_train) < n_splits:
        raise ValueError(
            f"Number of samples ({len(x_train)}) "
            f"must be greater than or equal to "
            f"n_splits ({n_splits})."
        )

    # --------------------------------------------------------
    # K-FOLD
    # --------------------------------------------------------

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # --------------------------------------------------------
    # SCORING
    #
    # sklearn returns negative values for loss metrics.
    # We convert RMSE and MAE back to positive values below.
    # --------------------------------------------------------

    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EXTRACT METRICS
    # --------------------------------------------------------

    r2_scores = scores["test_r2"]

    rmse_scores = -scores["test_rmse"]

    mae_scores = -scores["test_mae"]

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {
        "cv_r2_mean": float(
            r2_scores.mean()
        ),

        "cv_r2_std": float(
            r2_scores.std()
        ),

        "cv_rmse_mean": float(
            rmse_scores.mean()
        ),

        "cv_rmse_std": float(
            rmse_scores.std()
        ),

        "cv_mae_mean": float(
            mae_scores.mean()
        ),

        "cv_mae_std": float(
            mae_scores.std()
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
        """
        Train and return ONLY the fitted estimator.
        """
        raise NotImplementedError

    @abstractmethod
    def optimize_cv(
        self,
        trial,
        x_train,
        y_train,
        n_splits=DEFAULT_CV_FOLDS,
        random_state=DEFAULT_RANDOM_STATE,
    ):
        """
        Run one Optuna trial and return the CV R².
        """
        raise NotImplementedError


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

class RandomForestModel(Model):

    def train(
        self,
        x_train,
        y_train,
        **kwargs,
    ):
        """
        Train RandomForestRegressor.

        Returns:
            Fitted RandomForestRegressor.
        """

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
        """
        Optuna search space for Random Forest.
        """

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

        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf",
            1,
            10,
        )

        max_features = trial.suggest_categorical(
            "max_features",
            [
                "sqrt",
                "log2",
                1.0,
            ],
        )

        model = RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
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

        return float(
            metrics["cv_r2_mean"]
        )

    @staticmethod
    def _store_trial_metrics(
        trial,
        metrics,
    ):
        """
        Store additional CV metrics inside Optuna.
        """

        trial.set_user_attr(
            "cv_r2_std",
            float(
                metrics["cv_r2_std"]
            ),
        )

        trial.set_user_attr(
            "cv_rmse",
            float(
                metrics["cv_rmse_mean"]
            ),
        )

        trial.set_user_attr(
            "cv_rmse_std",
            float(
                metrics["cv_rmse_std"]
            ),
        )

        trial.set_user_attr(
            "cv_mae",
            float(
                metrics["cv_mae_mean"]
            ),
        )

        trial.set_user_attr(
            "cv_mae_std",
            float(
                metrics["cv_mae_std"]
            ),
        )


# ============================================================
# LIGHTGBM MODEL
# ============================================================

class LightGBMModel(Model):

    def train(
        self,
        x_train,
        y_train,
        **kwargs,
    ):
        """
        Train LGBMRegressor.

        Returns:
            Fitted LGBMRegressor.
        """

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
        """
        Optuna search space for LightGBM.
        """

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

        min_child_samples = trial.suggest_int(
            "min_child_samples",
            5,
            100,
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

        model = LGBMRegressor(
            random_state=random_state,
            verbosity=-1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
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

        return float(
            metrics["cv_r2_mean"]
        )

    @staticmethod
    def _store_trial_metrics(
        trial,
        metrics,
    ):
        """
        Store additional CV metrics inside Optuna.
        """

        trial.set_user_attr(
            "cv_r2_std",
            float(
                metrics["cv_r2_std"]
            ),
        )

        trial.set_user_attr(
            "cv_rmse",
            float(
                metrics["cv_rmse_mean"]
            ),
        )

        trial.set_user_attr(
            "cv_rmse_std",
            float(
                metrics["cv_rmse_std"]
            ),
        )

        trial.set_user_attr(
            "cv_mae",
            float(
                metrics["cv_mae_mean"]
            ),
        )

        trial.set_user_attr(
            "cv_mae_std",
            float(
                metrics["cv_mae_std"]
            ),
        )


# ============================================================
# XGBOOST MODEL
# ============================================================

class XGBoostModel(Model):

    def train(
        self,
        x_train,
        y_train,
        **kwargs,
    ):
        """
        Train XGBRegressor.

        Returns:
            Fitted XGBRegressor.
        """

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
        """
        Optuna search space for XGBoost.
        """

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

        min_child_weight = trial.suggest_int(
            "min_child_weight",
            1,
            10,
        )

        gamma = trial.suggest_float(
            "gamma",
            0.0,
            5.0,
        )

        reg_alpha = trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        )

        reg_lambda = trial.suggest_float(
            "reg_lambda",
            1e-8,
            10.0,
            log=True,
        )

        model = xgb.XGBRegressor(
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=-1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
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

        return float(
            metrics["cv_r2_mean"]
        )

    @staticmethod
    def _store_trial_metrics(
        trial,
        metrics,
    ):
        """
        Store additional CV metrics inside Optuna.
        """

        trial.set_user_attr(
            "cv_r2_std",
            float(
                metrics["cv_r2_std"]
            ),
        )

        trial.set_user_attr(
            "cv_rmse",
            float(
                metrics["cv_rmse_mean"]
            ),
        )

        trial.set_user_attr(
            "cv_rmse_std",
            float(
                metrics["cv_rmse_std"]
            ),
        )

        trial.set_user_attr(
            "cv_mae",
            float(
                metrics["cv_mae_mean"]
            ),
        )

        trial.set_user_attr(
            "cv_mae_std",
            float(
                metrics["cv_mae_std"]
            ),
        )


# ============================================================
# LINEAR REGRESSION MODEL
# ============================================================

class LinearRegressionModel(Model):

    def train(
        self,
        x_train,
        y_train,
        **kwargs,
    ):
        """
        Train LinearRegression.

        Returns:
            Fitted LinearRegression.
        """

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
        """
        Linear Regression has no meaningful
        hyperparameters to optimize for this pipeline.

        A trial is still executed so that it remains
        compatible with HyperparameterTuner.
        """

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

        return float(
            metrics["cv_r2_mean"]
        )

    @staticmethod
    def _store_trial_metrics(
        trial,
        metrics,
    ):
        """
        Store additional CV metrics inside Optuna.
        """

        trial.set_user_attr(
            "cv_r2_std",
            float(
                metrics["cv_r2_std"]
            ),
        )

        trial.set_user_attr(
            "cv_rmse",
            float(
                metrics["cv_rmse_mean"]
            ),
        )

        trial.set_user_attr(
            "cv_rmse_std",
            float(
                metrics["cv_rmse_std"]
            ),
        )

        trial.set_user_attr(
            "cv_mae",
            float(
                metrics["cv_mae_mean"]
            ),
        )

        trial.set_user_attr(
            "cv_mae_std",
            float(
                metrics["cv_mae_std"]
            ),
        )


# ============================================================
# OPTUNA HYPERPARAMETER TUNER
# ============================================================

class HyperparameterTuner:

    def __init__(
        self,
        model,
        x_train,
        y_train,
        n_splits=DEFAULT_CV_FOLDS,
        random_state=DEFAULT_RANDOM_STATE,
    ):
        """
        Initialize the hyperparameter tuner.
        """

        if not isinstance(
            model,
            Model,
        ):
            raise TypeError(
                "model must be an instance "
                "of the Model base class."
            )

        self.model = model

        self.x_train = x_train

        self.y_train = y_train

        self.n_splits = n_splits

        self.random_state = random_state

    def optimize(
        self,
        n_trials=30,
    ):
        """
        Optimize model hyperparameters using Optuna.

        Returns:
            Dictionary containing:
                - best parameters
                - best CV R²
                - CV standard deviations
                - RMSE
                - MAE
                - trial history
                - sampler
                - direction
        """

        if n_trials < 1:

            raise ValueError(
                "n_trials must be at least 1."
            )

        # ====================================================
        # 1. CREATE SAMPLER
        # ====================================================

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state,
        )

        # ====================================================
        # 2. CREATE STUDY
        # ====================================================

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        # ====================================================
        # 3. TRIAL CALLBACK
        # ====================================================

        def print_trial_result(
            study,
            trial,
        ):
            """
            Print the result of each completed trial.
            """

            if (
                trial.state
                != optuna.trial.TrialState.COMPLETE
            ):
                return

            if trial.value is None:
                return

            print(
                f"Trial "
                f"{trial.number + 1:02d}/"
                f"{n_trials} "
                f"| CV R²: "
                f"{trial.value:.6f} "
                f"| Best R²: "
                f"{study.best_value:.6f}"
            )

        # ====================================================
        # 4. OBJECTIVE
        # ====================================================

        def objective(trial):
            """
            Optuna objective function.
            """

            return self.model.optimize_cv(
                trial,
                self.x_train,
                self.y_train,
                n_splits=self.n_splits,
                random_state=self.random_state,
            )

        # ====================================================
        # 5. RUN OPTIMIZATION
        # ====================================================

        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[
                print_trial_result,
            ],
        )

        # ====================================================
        # 6. VALIDATE STUDY
        # ====================================================

        completed_trials = [
            trial
            for trial in study.trials
            if (
                trial.state
                == optuna.trial.TrialState.COMPLETE
            )
        ]

        if not completed_trials:

            raise RuntimeError(
                "Optuna completed no successful trials."
            )

        # ====================================================
        # 7. GET BEST TRIAL
        # ====================================================

        best_trial = study.best_trial

        best_params = dict(
            best_trial.params
        )

        # ====================================================
        # 8. RE-EVALUATE BEST PARAMETERS
        #
        # IMPORTANT:
        # We create a NEW estimator using the best params
        # and perform CV again.
        #
        # We do NOT fit the estimator before CV.
        # ====================================================

        best_metrics = (
            self._evaluate_best_parameters(
                best_params
            )
        )

        # ====================================================
        # 9. BUILD TRIAL HISTORY
        # ====================================================

        trial_history = []

        for trial in study.trials:

            duration_seconds = None

            if trial.duration is not None:

                duration_seconds = float(
                    trial.duration.total_seconds()
                )

            trial_record = {
                "trial_number": int(
                    trial.number
                ),

                "state": (
                    trial.state.name
                ),

                "value": (
                    float(trial.value)
                    if trial.value is not None
                    else None
                ),

                "params": dict(
                    trial.params
                ),

                "cv_r2_std": self._get_trial_attr(
                    trial,
                    "cv_r2_std",
                ),

                "cv_rmse": self._get_trial_attr(
                    trial,
                    "cv_rmse",
                ),

                "cv_rmse_std": self._get_trial_attr(
                    trial,
                    "cv_rmse_std",
                ),

                "cv_mae": self._get_trial_attr(
                    trial,
                    "cv_mae",
                ),

                "cv_mae_std": self._get_trial_attr(
                    trial,
                    "cv_mae_std",
                ),

                "duration_seconds": (
                    duration_seconds
                ),
            }

            trial_history.append(
                trial_record
            )

        # ====================================================
        # 10. FINAL REPORT
        # ====================================================

        print("\n" + "=" * 60)
        print("OPTUNA FINAL RESULTS")
        print("=" * 60)

        print(
            f"Best Trial      : "
            f"{best_trial.number + 1}"
        )

        print(
            f"Best CV R²      : "
            f"{best_metrics['cv_r2_mean']:.6f}"
        )

        print(
            f"CV R² Std       : "
            f"{best_metrics['cv_r2_std']:.6f}"
        )

        print(
            f"CV RMSE         : "
            f"{best_metrics['cv_rmse_mean']:.6f}"
        )

        print(
            f"CV RMSE Std     : "
            f"{best_metrics['cv_rmse_std']:.6f}"
        )

        print(
            f"CV MAE          : "
            f"{best_metrics['cv_mae_mean']:.6f}"
        )

        print(
            f"CV MAE Std      : "
            f"{best_metrics['cv_mae_std']:.6f}"
        )

        print(
            f"Best Parameters : "
            f"{best_params}"
        )

        print("=" * 60)

        # ====================================================
        # 11. RETURN SERIALIZABLE RESULT
        # ====================================================

        return {
            "best_trial_number": int(
                best_trial.number
            ),

            "best_params": best_params,

            "optuna_best_value": float(
                best_trial.value
            ),

            "best_cv_r2": float(
                best_metrics["cv_r2_mean"]
            ),

            "cv_r2_std": float(
                best_metrics["cv_r2_std"]
            ),

            "cv_rmse": float(
                best_metrics["cv_rmse_mean"]
            ),

            "cv_rmse_std": float(
                best_metrics["cv_rmse_std"]
            ),

            "cv_mae": float(
                best_metrics["cv_mae_mean"]
            ),

            "cv_mae_std": float(
                best_metrics["cv_mae_std"]
            ),

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

            "trial_history": trial_history,
        }

    def _evaluate_best_parameters(
        self,
        best_params,
    ):
        """
        Evaluate the best Optuna parameters using
        fresh K-Fold CV.

        The model is NOT pre-fitted before cross_validate().
        """

        model_name = (
            self.model.__class__.__name__
        )

        if model_name == "RandomForestModel":

            estimator = RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                **best_params,
            )

        elif model_name == "LightGBMModel":

            estimator = LGBMRegressor(
                random_state=self.random_state,
                verbosity=-1,
                **best_params,
            )

        elif model_name == "XGBoostModel":

            estimator = xgb.XGBRegressor(
                random_state=self.random_state,
                objective="reg:squarederror",
                n_jobs=-1,
                **best_params,
            )

        elif model_name == "LinearRegressionModel":

            estimator = LinearRegression(
                **best_params,
            )

        else:

            raise ValueError(
                f"Unsupported model wrapper: "
                f"{model_name}"
            )

        return evaluate_model_cv(
            estimator,
            self.x_train,
            self.y_train,
            n_splits=self.n_splits,
            random_state=self.random_state,
        )

    @staticmethod
    def _get_trial_attr(
        trial,
        key,
    ):
        """
        Safely retrieve an Optuna user attribute.
        """

        value = trial.user_attrs.get(
            key
        )

        if value is None:
            return None

        return float(value)


# ============================================================
# MODEL BENCHMARK
# ============================================================

class ModelBenchmark:

    def __init__(
        self,
        n_splits=DEFAULT_CV_FOLDS,
        random_state=DEFAULT_RANDOM_STATE,
    ):
        """
        Initialize model benchmark.
        """

        if n_splits < 2:

            raise ValueError(
                "n_splits must be at least 2."
            )

        self.n_splits = n_splits

        self.random_state = random_state

    def compare_models(
        self,
        models,
        x_train,
        y_train,
    ):
        """
        Compare candidate models using identical
        K-Fold CV configuration.

        Args:
            models:
                Dictionary of model_name -> estimator.

        Returns:
            DataFrame sorted by CV R² descending.
        """

        if not models:

            raise ValueError(
                "No models were provided "
                "for benchmarking."
            )

        results = []

        for model_name, model in models.items():

            print(
                f"\nBenchmarking "
                f"{model_name}..."
            )

            metrics = evaluate_model_cv(
                model,
                x_train,
                y_train,
                n_splits=self.n_splits,
                random_state=self.random_state,
            )

            mean_r2 = float(
                metrics["cv_r2_mean"]
            )

            std_r2 = float(
                metrics["cv_r2_std"]
            )

            mean_rmse = float(
                metrics["cv_rmse_mean"]
            )

            std_rmse = float(
                metrics["cv_rmse_std"]
            )

            mean_mae = float(
                metrics["cv_mae_mean"]
            )

            std_mae = float(
                metrics["cv_mae_std"]
            )

            print(
                f"{model_name} → "
                f"R²={mean_r2:.6f} "
                f"± {std_r2:.6f}, "
                f"RMSE={mean_rmse:.6f} "
                f"± {std_rmse:.6f}, "
                f"MAE={mean_mae:.6f} "
                f"± {std_mae:.6f}"
            )

            results.append(
                {
                    "model": str(
                        model_name
                    ),

                    "cv_r2": mean_r2,

                    "cv_r2_std": std_r2,

                    "cv_rmse": mean_rmse,

                    "cv_rmse_std": std_rmse,

                    "cv_mae": mean_mae,

                    "cv_mae_std": std_mae,
                }
            )

        # ====================================================
        # BUILD DATAFRAME
        # ====================================================

        results_df = pd.DataFrame(
            results
        )

        if results_df.empty:

            raise RuntimeError(
                "Model benchmarking produced "
                "no results."
            )

        # ====================================================
        # SORT BY CV R²
        # ====================================================

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
    """
    Return the baseline candidate estimators.

    IMPORTANT:
    These are UNFITTED estimators.

    ModelBenchmark will clone/fits them through
    sklearn.cross_validate().
    """

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