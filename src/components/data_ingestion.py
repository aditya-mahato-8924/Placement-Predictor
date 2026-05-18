import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_params, get_root_directory
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(get_root_directory(), "data/train_data/train.csv")
    test_data_path: str = os.path.join(get_root_directory(), "data/test_data/test.csv")
    raw_data_path: str = os.path.join("notebooks/data", "student_placement.csv")
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # read the data from the path
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Read the dataset as dataframe from {self.ingestion_config.raw_data_path}")

            # load the params from yaml
            PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml')
            params = load_params(params_path=PARAMS_PATH)
            logging.info(f'Parameters retrieved from {PARAMS_PATH}')
            
            test_size = params['data_ingestion']['test_size']

            # perform train test split
            train_set, test_set = train_test_split(df, test_size=test_size, random_state=42, stratify=df['placement_status'])

            # create the directories of the train_data and test_data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            # save the train and test data into their respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Ingestion of the data is completed")

        except Exception as e:
            logging.error("Error occurred in data ingestion: %s", str(e))
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()