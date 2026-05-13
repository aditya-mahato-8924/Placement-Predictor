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

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

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
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
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

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor_obj)

            logging.info("Saved preprocessing object")
            logging.info("Data transformation is completed")

            return (
                train_arr,
                test_arr
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        