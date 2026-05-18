import os
import sys
from dataclasses import dataclass
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import get_root_directory

@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join(get_root_directory(), "data/train_data/train.csv")
    test_data_path: str = os.path.join(get_root_directory(), "data/test_data/test.csv")
    preprocessor_obj_file_path: str = os.path.join(get_root_directory(), "preprocessor/preprocessor.pkl")
    preprocessed_train_data_path: str = os.path.join(get_root_directory(), "data/preprocessed/train.npy")
    preprocessed_test_data_path: str = os.path.join(get_root_directory(), "data/preprocessed/test.npy")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """This function is responsible for data transformation
        """
        try:
            logging.info("Data Transformation initiated")

            categorical_cols = ['branch', 'college_tier']
            numerical_cols = ['cgpa', 'backlogs', 'coding_skills', 'dsa_score', 'aptitude_score', 'communication_skills', 'ml_knowledge', 'system_design', 'internships', 'projects_count', 'certifications', 'hackathons', 'open_source_contributions', 'extracurriculars']

            num_pipeline = Pipeline(steps=[
                ('scaler', RobustScaler())
            ])

            cat_pipeline1 = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])

            cat_pipeline2 = Pipeline(steps=[
                ('ordinal', OrdinalEncoder(categories=[['Tier-1', 'Tier-2', 'Tier-3']], handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            logging.info(f"Categorical Columns:{categorical_cols}")
            logging.info(f"Numerical Columns:{numerical_cols}")

            preprocessor = ColumnTransformer(transformers=[
                ('cat_pipeline1', cat_pipeline1, [categorical_cols[0]]),
                ('cat_pipeline2', cat_pipeline2, [categorical_cols[1]]),
                ('num_pipeline', num_pipeline, numerical_cols),
            ])

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self):
        try:
            # read the train and test data
            train_df = pd.read_csv(self.data_transformation_config.train_data_path)
            test_df = pd.read_csv(self.data_transformation_config.test_data_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            # fetch the preprocessor object
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'placement_status'

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframe")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # create the directories to save train and test array
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessed_train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessed_test_data_path), exist_ok=True)

            # save the train_arr and test_arr
            np.save(
                file = self.data_transformation_config.preprocessed_train_data_path,
                arr = train_arr
            )

            np.save(
                file = self.data_transformation_config.preprocessed_test_data_path,
                arr = test_arr
            )

            # save the preprocessor object
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor_obj)

            logging.info("Saved preprocessing object")
            logging.info("Data transformation is completed")

        except Exception as e:
            logging.error(f"Failed to Complete data transformation : {e}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation()