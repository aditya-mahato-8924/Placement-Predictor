import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Logistic Regression': LogisticRegression(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Support Vector Machine': LinearSVC(),
                'Random Forest': RandomForestClassifier(),
                'XGBoost': XGBClassifier()
            }

            param_grids = {
                'Logistic Regression': {
                        'l1_ratio': [0, 1],
                        'solver': ['liblinear'],
                        'C': [0.01, 0.1, 1, 10],
                        'class_weight': ['balanced', {0: 1.5, 1:1}, {0:2, 1:1}, {0:1.25, 1:1}],
                        'max_iter': [100, 500, 1000]
                    },
                
                'K-Nearest Neighbors': {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan', 'minkowski']
                    },
                
                'Support Vector Machine': {
                    'penalty': ['l1'],
                    'loss': ['squared_hinge'],
                    'C': [0.1, 0.5, 1, 10],
                    'class_weight': [
                        'balanced',
                        {0:1.5, 1:1},
                        {0:2, 1:1}
                    ],
                    'max_iter': [10000, 5000, 1000]
                },

                'Random Forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [3, 5, 10, 20],
                    'class_weight': ['balanced', {0: 1.5, 1:1}, {0:2, 1:1}, {0:1.25, 1:1}]
                },

                'XGBoost': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 10, 15],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.5, 1, 10],
                    'reg_lambda': [0, 0.5, 1, 5]
                }
                
            }

            logging.info('Model Training initiated')

            model_report, models_dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param_grids=param_grids)

            logging.info("Model Training completed")

            best_model_name, best_score = None, float('-inf')

            for model_name, metrics in model_report.items():
                if metrics['test_auc'] > best_score:
                    best_score = metrics['test_auc']
                    best_model_name = model_name
                
            best_model = models_dict[best_model_name]

            logging.info(f'Best model found on both training and testing dataset, model name: {best_model_name}, AUC-ROC: {best_score}')
            
            print(f'Best model found, model name: {best_model_name}, AUC-ROC: {best_score}')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info(f"Trained model is saved at path: {self.model_trainer_config.trained_model_file_path}")
            print(model_report)

        except Exception as e:
            raise CustomException(e, sys)