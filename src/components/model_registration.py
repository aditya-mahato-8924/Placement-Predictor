import os
import sys
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from src.logger import logging
from src.exception import CustomException
from src.utils import get_root_directory, load_model_info

import dagshub
dagshub.init(repo_owner='aditya-mahato-8924', repo_name='Placement-Predictor', mlflow=True)


@dataclass
class ModelRegistrationConfig:
    model_info_path: str = os.path.join(
        get_root_directory(),
        "model/model_info.json"
    )


class ModelRegistration:

    def __init__(self):
        self.model_registration_config = ModelRegistrationConfig()

    def initiate_model_registration(self):

        try:
            logging.info("Starting model registration process")

            # initialize mlflow client
            client = MlflowClient()

            # load experiment/model information
            logging.info("Loading experiment information")

            model_info = load_model_info(
                self.model_registration_config.model_info_path
            )

            # create model uri
            model_uri = model_info["model_uri"]

            logging.info(f"Model URI: {model_uri}")

            # model name in registry
            model_name = "Placement Predictor Model"

            # register model
            logging.info("Registering model to MLflow Model Registry")

            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )

            model_version = registered_model.version

            logging.info(
                f"Model registered successfully "
                f"with version: {model_version}"
            )

            # set alias instead of deprecated stages
            logging.info("Setting model alias as 'champion'")

            client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=model_version
            )

            logging.info(
                f"Alias 'champion' assigned to "
                f"model version {model_version}"
            )

            logging.info("Model registration completed successfully")

        except Exception as e:

            logging.error(
                f"Failed to complete model registration: {e}"
            )

            raise CustomException(e, sys)


if __name__ == "__main__":
    model_registration = ModelRegistration()
    model_registration.initiate_model_registration()