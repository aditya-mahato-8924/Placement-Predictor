import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, get_root_directory, evaluate_model, save_model_info

import dagshub
dagshub.init(repo_owner='aditya-mahato-8924', repo_name='Placement-Predictor', mlflow=True)

@dataclass
class ModelEvaluationConfig:
    preprocessed_test_data_path: str = os.path.join(get_root_directory(), "data/preprocessed/test.npy")
    model_obj_path: str = os.path.join(get_root_directory(), "model/model.pkl")
    confusion_matrix_path: str = os.path.join(get_root_directory(), "model/metrics/confusion_matrix.png")
    preprocessor_obj_path: str = os.path.join(get_root_directory(), "preprocessor/preprocessor.pkl")
    model_info_path: str = os.path.join(get_root_directory(), "model/model_info.json")

class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()

    def initiate_model_evaluation(self):
        try:
            mlflow.set_tracking_uri(
                "https://dagshub.com/aditya-mahato-8924/Placement-Predictor.mlflow"
            )

            mlflow.set_experiment("Placement Predictor Final Model")

            with mlflow.start_run(run_name="Final Model Evaluation") as run:

                # load test data
                test_array = np.load(self.model_evaluation_config.preprocessed_test_data_path)
                logging.info(f"Test array loaded from {self.model_evaluation_config.preprocessed_test_data_path}")

                X_test, y_test = test_array[:, :-1], test_array[:, -1]

                # load model
                model = load_object(self.model_evaluation_config.model_obj_path)
                logging.info(f"Model loaded from {self.model_evaluation_config.model_obj_path}")

                # predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                
                # evaluate and log the metrics
                metrics = evaluate_model(y_test, y_pred, y_prob)
                mlflow.log_metrics(metrics)
                logging.info("Metrics logged to MLflow")

                
                # log the confusion matrix
                fig = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

                # create a folder metrics inside model
                os.makedirs(os.path.dirname(self.model_evaluation_config.confusion_matrix_path), exist_ok=True)
                fig.figure_.savefig(self.model_evaluation_config.confusion_matrix_path)
                plt.close()

                mlflow.log_artifact(self.model_evaluation_config.confusion_matrix_path)
                logging.info("Confusion matrix saved and logged")

                # log the params
                model_params = model.get_params()
                mlflow.log_params({
                    'C' : model_params['C'],
                    'l1_ratio': model_params['l1_ratio'],
                    'solver': model_params['solver'],
                    'max_iter': model_params['max_iter'],
                    'class_weight': model_params['class_weight']
                })
                
                # log model
                logged_model = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model"
                )

                # save the model info
                model_info = {
                    "model_uri": logged_model.model_uri,
                    "model_id": logged_model.model_id,
                    "run_id": logged_model.run_id
                }

                save_model_info(model_info, self.model_evaluation_config.model_info_path)
                logging.info(f"Model info saved to {self.model_evaluation_config.model_info_path}")

                # log the preprocessor in the artifact
                mlflow.log_artifact(self.model_evaluation_config.preprocessor_obj_path, artifact_path="preprocessor")
                logging.info("Preprocessor logged to Mlflow")

                # set the tags
                mlflow.set_tags(
                    {
                        'model_type': 'final_model',
                        'model_type': 'Logistic Regression',
                        'dataset_version': 'v1'
                    }
                )

        except Exception as e:
            logging.error(f"Failed to Complete model evaluation : {e}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    model_evaluation = ModelEvaluation()
    model_evaluation.initiate_model_evaluation()