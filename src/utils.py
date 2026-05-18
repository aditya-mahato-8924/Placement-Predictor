import os
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import yaml
import json

def save_object(file_path, obj):
    """Saves a Python object to a file using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path) -> object:
    """Loads a Python object from a file using pickle."""
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_params(params_path:str) -> dict:
    """Load parameters from yaml file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        raise CustomException(e, sys)

def get_root_directory():
    """Return the root directory path"""

    root_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".."
    )

    return root_dir

def evaluate_model(y_true, y_pred, y_prob):
    """Evaluate the performance of machine learning model"""
    
    accuracy = accuracy_score(y_true, y_pred)
    precision_1 = precision_score(y_true, y_pred)
    recall_1 = recall_score(y_true, y_pred)
    f1_1 = f1_score(y_true, y_pred)

    precision_0 = precision_score(y_true, y_pred, pos_label=0)
    recall_0 = recall_score(y_true, y_pred, pos_label=0)
    f1_0 = f1_score(y_true, y_pred, pos_label=0)
    auc_score = roc_auc_score(y_true, y_prob)
 
    return {
        'accuracy': accuracy,
        'precision_0': precision_0,
        'recall_0': recall_0,
        'f1_0': f1_0,
        'precision_1': precision_1,
        'recall_1': recall_1,
        'f1_1': f1_1,
        'roc_auc_score': auc_score
    }

def save_model_info(model_info:dict, file_path:str) -> None:
    """Save the model info into the specified path in JSON format."""

    # save the dictionary as a JSON file
    with open(file_path, 'w') as file:
        json.dump(model_info, file, indent=4)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    with open(file_path, 'r') as file:
        model_info = json.load(file)
    return model_info