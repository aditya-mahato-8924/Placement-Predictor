import sys
import pandas as pd
from src.utils import load_object
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data: pd.DataFrame) -> dict:
        """Predict the placement outcome based on input features."""
        try:
            # load the preprocessor and model
            MODEL_PATH = 'artifacts/model.pkl'
            PREPROCESSOR_PATH = 'artifacts/preprocessor.pkl'

            model = load_object(MODEL_PATH)
            preprocessor = load_object(PREPROCESSOR_PATH)

            # preprocess the input data
            preprocessed_data = preprocessor.transform(data)

            # make predictions
            prediction = int(model.predict(preprocessed_data)[0])
            probability = round(model.predict_proba(preprocessed_data)[0][prediction], 2)

            feature_names = preprocessor.get_feature_names_out()
            feature_contributions = {
                feature: float(round(coef, 2)) for feature, coef in zip(feature_names, model.coef_[0])
            }

            print(prediction)
            print(probability)
            print(feature_contributions)

            return {'prediction': prediction, 'probability': probability, 'feature_contributions': feature_contributions}
                 
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
            self,
            branch: str,
            college_tier: str,
            cgpa: float,
            backlogs: int,
            coding_skills: float,
            dsa_score: float,
            aptitude_score: float,
            communication_skills: float,
            ml_knowledge: float,
            system_design: float,
            internships: int,
            projects_count: int,
            certifications: int,
            hackathons: int,
            open_source_contributions: int,
            extracurriculars: int
    ):
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

    def get_data_as_data_frame(self):
        """Convert the input data to a pandas dataframe."""
        try:
            custom_data_input_dict = {
                'branch': [self.branch],
                'college_tier': [self.college_tier],
                'cgpa': [self.cgpa],
                'backlogs': [self.backlogs],
                'coding_skills': [self.coding_skills],
                'dsa_score': [self.dsa_score],
                'aptitude_score': [self.aptitude_score],
                'communication_skills': [self.communication_skills],
                'ml_knowledge': [self.ml_knowledge],
                'system_design': [self.system_design],
                'internships': [self.internships],
                'projects_count': [self.projects_count],
                'certifications': [self.certifications],
                'hackathons': [self.hackathons],
                'open_source_contributions': [self.open_source_contributions],
                'extracurriculars': [self.extracurriculars]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)