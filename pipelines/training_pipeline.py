from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation
from steps.config import ModelNameConfig


@pipeline(enable_cache=True)
def train_pipeline():

    # 1. Ingest data
    df = ingest_data()

    # 2. Clean and split data
    x_train, x_test, y_train, y_test = clean_data(df)

    # 3. Model configuration
    config = ModelNameConfig(
        model_name="lightgbm",
        fine_tuning=False,
    )

    # 4. Train model
    model = train_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        config=config,
    )

    # 5. Evaluate model
    mse, rmse = evaluation(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )