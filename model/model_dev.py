import optuna
import pandas as pd
import xgboost as xgb

from abc import ABC, abstractmethod
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate


# ============================================================
# BASE MODEL
# ============================================================

class Model(ABC):

    @abstractmethod
    def train(self, x_train, y_train, **kwargs):
        pass

    @abstractmethod
    def optimize(
        self,
        trial,
        x_train,
        y_train,
        x_valid,
        y_valid,
    ):
        pass


# ============================================================
# RANDOM FOREST
# ============================================================

class RandomForestModel(Model):

    def train(self, x_train, y_train, **kwargs):

        reg = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **kwargs,
        )

        reg.fit(x_train, y_train)

        return reg

    def optimize(
        self,
        trial,
        x_train,
        y_train,
        x_valid,
        y_valid,
    ):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                500,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                5,
                30,
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                20,
            ),
        }

        reg = self.train(
            x_train,
            y_train,
            **params,
        )

        return reg.score(
            x_valid,
            y_valid,
        )


# ============================================================
# LIGHTGBM
# ============================================================

class LightGBMModel(Model):

    def train(self, x_train, y_train, **kwargs):

        reg = LGBMRegressor(
            random_state=42,
            verbosity=-1,
            **kwargs,
        )

        reg.fit(x_train, y_train)

        return reg

    def optimize(
        self,
        trial,
        x_train,
        y_train,
        x_valid,
        y_valid,
    ):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                1000,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                15,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                20,
                150,
            ),
        }

        reg = self.train(
            x_train,
            y_train,
            **params,
        )

        return reg.score(
            x_valid,
            y_valid,
        )


# ============================================================
# XGBOOST
# ============================================================

class XGBoostModel(Model):

    def train(self, x_train, y_train, **kwargs):

        reg = xgb.XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
            **kwargs,
        )

        reg.fit(x_train, y_train)

        return reg

    def optimize(
        self,
        trial,
        x_train,
        y_train,
        x_valid,
        y_valid,
    ):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                1000,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),
        }

        reg = self.train(
            x_train,
            y_train,
            **params,
        )

        return reg.score(
            x_valid,
            y_valid,
        )


# ============================================================
# LINEAR REGRESSION
# ============================================================

class LinearRegressionModel(Model):

    def train(self, x_train, y_train, **kwargs):

        reg = LinearRegression(
            **kwargs,
        )

        reg.fit(
            x_train,
            y_train,
        )

        return reg

    def optimize(
        self,
        trial,
        x_train,
        y_train,
        x_valid,
        y_valid,
    ):

        reg = self.train(
            x_train,
            y_train,
        )

        return reg.score(
            x_valid,
            y_valid,
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
        x_valid,
        y_valid,
    ):

        self.model = model

        self.x_train = x_train
        self.y_train = y_train

        self.x_valid = x_valid
        self.y_valid = y_valid

    def optimize(
        self,
        n_trials=10,
    ):

        study = optuna.create_study(
            direction="maximize",
        )

        study.optimize(
            lambda trial:
                self.model.optimize(
                    trial,
                    self.x_train,
                    self.y_train,
                    self.x_valid,
                    self.y_valid,
                ),
            n_trials=n_trials,
        )

        return study.best_trial.params


# ============================================================
# MODEL BENCHMARK
# ============================================================

class ModelBenchmark:

    def __init__(
        self,
        n_splits=5,
        random_state=42,
    ):

        self.n_splits = n_splits
        self.random_state = random_state

    def compare_models(
        self,
        models,
        x_train,
        y_train,
    ):

        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        scoring = {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        }

        results = []

        for model_name, model in models.items():

            print(
                f"\nBenchmarking {model_name}..."
            )

            scores = cross_validate(
                clone(model),
                x_train,
                y_train,
                cv=kfold,
                scoring=scoring,
                n_jobs=-1,
            )

            mean_r2 = scores[
                "test_r2"
            ].mean()

            mean_rmse = -scores[
                "test_rmse"
            ].mean()

            mean_mae = -scores[
                "test_mae"
            ].mean()

            std_r2 = scores[
                "test_r2"
            ].std()

            print(
                f"{model_name} → "
                f"R²={mean_r2:.4f}, "
                f"RMSE={mean_rmse:.4f}, "
                f"MAE={mean_mae:.4f}"
            )

            results.append(
                {
                    "model": model_name,
                    "cv_r2": mean_r2,
                    "cv_r2_std": std_r2,
                    "cv_rmse": mean_rmse,
                    "cv_mae": mean_mae,
                }
            )

        results_df = pd.DataFrame(
            results
        )

        results_df = results_df.sort_values(
            by="cv_r2",
            ascending=False,
        ).reset_index(
            drop=True
        )

        return results_df


# ============================================================
# CANDIDATE MODELS
# ============================================================

def get_candidate_models():

    return {

        "lightgbm": LGBMRegressor(
            random_state=42,
            verbosity=-1,
        ),

        "xgboost": xgb.XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
        ),

        "randomforest": RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        ),

        "linear_regression": LinearRegression(),

    }