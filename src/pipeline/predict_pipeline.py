import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import get_root_directory, load_object

class CustomData:
    def __init__(
        self,
        branch: str,
        college_tier: int,
        cgpa: float,
        backlogs: int,
        coding_skills: int,
        dsa_score: int,
        aptitude_score: int,
        communication_skills: int,
        ml_knowledge: int,
        system_design: int,
        internships: int,
        projects_count: int,
        certifications: int,
        hackathons: int,
        open_source_contributions: int,
        extracurriculars: int ):

        self.branch = branch
        self.college_tier = college_tier
        self.cgpa = cgpa
        self.backlogs = backlogs
        self.coding_skills = coding_skills
        self.dsa_score = dsa_score
        self.aptitude_score = aptitude_score
        self.communication_skills = communication_skills
        self.ml_knowledge = ml_knowledge
        self.system_design = system_design
        self.internships = internships
        self.projects_count = projects_count
        self.certifications = certifications
        self.hackathons = hackathons
        self.open_source_contributions = open_source_contributions
        self.extracurriculars = extracurriculars

    def get_data_as_dataframe(self) -> pd.DataFrame:

        try:
            custom_data_input_dict = {
                "branch": [self.branch],
                "college_tier": [self.college_tier],
                "cgpa": [self.cgpa],
                "backlogs": [self.backlogs],
                "coding_skills": [self.coding_skills],
                "dsa_score": [self.dsa_score],
                "aptitude_score": [self.aptitude_score],
                "communication_skills": [self.communication_skills],
                "ml_knowledge": [self.ml_knowledge],
                "system_design": [self.system_design],
                "internships": [self.internships],
                "projects_count": [self.projects_count],
                "certifications": [self.certifications],
                "hackathons": [self.hackathons],
                "open_source_contributions": [self.open_source_contributions],
                "extracurriculars": [self.extracurriculars]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


class PredictPipelineConfig:
    def __init__(self):

        self.model_path = os.path.join(
            get_root_directory(),
            "model",
            "model.pkl"
        )

        self.preprocessor_path = os.path.join(
            get_root_directory(),
            "preprocessor",
            "preprocessor.pkl"
        )


class PredictPipeline:

    def __init__(self):
        self.predict_pipeline_config = PredictPipelineConfig()

    def get_prediction(self, input_df: pd.DataFrame) -> dict:
        logging.info("Prediction Pipeline Initiated ...")

        try:
            # load artifacts
            preprocessor = load_object(
                self.predict_pipeline_config.preprocessor_path
            )

            model = load_object(
                self.predict_pipeline_config.model_path
            )

            logging.info("Model and Preprocessor loaded successfully !!!")

            # transform input data
            transformed_data = preprocessor.transform(input_df)

            logging.info("Data transformation done successfully !!!")

            # prediction
            predicted_category = int(
                model.predict(transformed_data)[0]
            )

            # probability
            probability = float(
                round(
                    model.predict_proba(transformed_data)[0][predicted_category],
                    2
                )
            )

            logging.info("Model Prediction done successfully !!!")

            result = {
                "predicted_category": predicted_category,
                "probability": probability
            }

            logging.info("Prediction Pipeline completed successfully !!!")

            return result

        except Exception as e:
            logging.error(f"Failed to complete the prediction pipeline : {e}")
            raise CustomException(e, sys)