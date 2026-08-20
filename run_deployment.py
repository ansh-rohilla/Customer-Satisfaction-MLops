from typing import cast

import click
from rich import print

from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService


DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice(
        [DEPLOY, PREDICT, DEPLOY_AND_PREDICT]
    ),
    default=DEPLOY_AND_PREDICT,
    help=(
        "Choose whether to deploy, predict, or both. "
        "Default: deploy_and_predict."
    ),
)
@click.option(
    "--min-r2",
    default=0.08,
    help="Minimum R² score required to deploy the model",
)
def main(config: str, min_r2: float):

    mlflow_model_deployer_component = (
        MLFlowModelDeployer.get_active_model_deployer()
    )

    deploy = config in [DEPLOY, DEPLOY_AND_PREDICT]
    predict = config in [PREDICT, DEPLOY_AND_PREDICT]

    if deploy:
        continuous_deployment_pipeline(
            min_r2=min_r2,
            workers=3,
            timeout=60,
        )

    if predict:
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )

    print(
        "\nYou can run:\n"
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "\n"
        "Use the MLflow UI to inspect and compare your experiment runs.\n"
    )

    existing_services = (
        mlflow_model_deployer_component.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name="model",
        )
    )

    if existing_services:
        service = cast(
            MLFlowDeploymentService,
            existing_services[0],
        )

        if service.is_running:
            print(
                "\nThe MLflow prediction server is running.\n"
                f"Prediction URL: {service.prediction_url}\n"
            )

        elif service.is_failed:
            print(
                "\nThe MLflow prediction server failed.\n"
                f"Last state: {service.status.state.value}\n"
                f"Last error: {service.status.last_error}\n"
            )

    else:
        print(
            "\nNo MLflow prediction server is currently running.\n"
            "Run the deployment pipeline first."
        )


if __name__ == "__main__":
    main()