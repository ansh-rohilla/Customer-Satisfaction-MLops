from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


if __name__ == "__main__":
    training = train_pipeline()

    print(
        "\nNow run:\n"
        f"mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "\nTo inspect your experiment runs within the MLflow UI."
    )