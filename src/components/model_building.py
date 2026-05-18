import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.exception import CustomException
from src.logger import logging
from src.utils import load_params

from src.utils import save_object, get_root_directory

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(get_root_directory(), 'model/model.pkl')
    train_arr_path: str = os.path.join(get_root_directory(), "data/preprocessed/train.npy")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self):
        try:
            # load the train array
            train_array = np.load(self.model_trainer_config.train_arr_path)

            logging.info("Train array loaded")
            logging.info("Split the target column from train array")

            X_train, y_train = (
                train_array[:, :-1],
                train_array[:, -1]
            )

            # load the model parameters from params.yaml
            PARAMS_PATH = os.path.join(get_root_directory(), 'params.yaml')
            params = load_params(params_path=PARAMS_PATH)
            logging.info(f'Model Parameters retrieved from {PARAMS_PATH}')

        
            model = LogisticRegression(
                C = params['model_building']['C'],
                solver = params['model_building']['solver'],
                l1_ratio = params['model_building']['l1_ratio'],
                max_iter = params['model_building']['max_iter'],
                class_weight = params['model_building']['class_weight']
            )

            # fit the data into model
            model.fit(X_train, y_train)

            logging.info("Model Training completed")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = model
            )

            logging.info(f"Trained model is saved at path: {self.model_trainer_config.trained_model_file_path}")

        except Exception as e:
            logging.error(f"Failed to Complete model building : {e}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer()