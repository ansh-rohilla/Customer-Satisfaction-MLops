import logging

import mlflow
import pandas as pd

from model.evaluation import MSE, RMSE, R2Score

from sklearn.base import RegressorMixin

from typing import Tuple
from typing_extensions import Annotated

from zenml import step
from zenml.client import Client


# ============================================================
# MLFLOW EXPERIMENT TRACKER
# ============================================================

experiment_tracker = (
    Client()
    .active_stack
    .experiment_tracker
)


# ============================================================
# EVALUATION STEP
# ============================================================

@step(
    experiment_tracker=experiment_tracker.name,
)
def evaluation(
    model: RegressorMixin,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
    Annotated[float, "mse"],
]:
    """
    Evaluate the trained model on the held-out test set.

    Metrics:
        - R²
        - RMSE
        - MSE

    Returns:
        r2_score: Test R² score
        rmse: Test RMSE
        mse: Test MSE
    """

    try:

        # ====================================================
        # 1. GENERATE TEST PREDICTIONS
        # ====================================================

        prediction = model.predict(x_test)

        # ====================================================
        # 2. CALCULATE TEST MSE
        # ====================================================

        mse_class = MSE()

        mse = float(
            mse_class.calculate_score(
                y_test,
                prediction,
            )
        )

        # ====================================================
        # 3. CALCULATE TEST R²
        # ====================================================

        r2_class = R2Score()

        r2_score = float(
            r2_class.calculate_score(
                y_test,
                prediction,
            )
        )

        # ====================================================
        # 4. CALCULATE TEST RMSE
        # ====================================================

        rmse_class = RMSE()

        rmse = float(
            rmse_class.calculate_score(
                y_test,
                prediction,
            )
        )

        # ====================================================
        # 5. LOG FINAL TEST METRICS TO MLFLOW
        # ====================================================

        mlflow.log_metric(
            "test_r2",
            r2_score,
        )

        mlflow.log_metric(
            "test_rmse",
            rmse,
        )

        mlflow.log_metric(
            "test_mse",
            mse,
        )

        # ====================================================
        # 6. DISPLAY FINAL TEST RESULTS
        # ====================================================

        print("\n" + "=" * 60)
        print("FINAL TEST SET EVALUATION")
        print("=" * 60)

        print(
            f"Test R²   : {r2_score:.6f}"
        )

        print(
            f"Test RMSE : {rmse:.6f}"
        )

        print(
            f"Test MSE  : {mse:.6f}"
        )

        print("=" * 60)

        # ====================================================
        # 7. RETURN METRICS
        # ====================================================
        #
        # IMPORTANT:
        # The order MUST match deployment_pipeline.py:
        #
        #     r2, rmse, mse = evaluation(...)
        #
        # ====================================================

        return (
            r2_score,
            rmse,
            mse,
        )

    except Exception as e:

        logging.error(
            f"Model evaluation failed: {e}"
        )

        raise