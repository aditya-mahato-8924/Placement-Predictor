from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        logging.info("Training pipeline initiated")
        # Step 1: Data Ingestion
        train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        train_array, test_array = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # Step 3: Model Training
        self.model_trainer.initiate_model_trainer(train_array, test_array)

        logging.info("Training pipeline completed successfully")

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()