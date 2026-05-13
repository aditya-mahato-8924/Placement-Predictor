import os
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

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

def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids):
    """Evaluates multiple machine learning models and returns a report of their performance."""
    try:
        model_report = {}
        models_dict = {}
        for model_name, model in models.items():
            param_grid = param_grids[model_name]

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            # perform hyperparameter tuning using RandomizedSearchCV
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=20, cv=cv, verbose=0, scoring='roc_auc', n_jobs=-1)

            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_

            # make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # probability predictions for AUC-ROC
            if hasattr(best_model, "predict_proba"):
                y_train_proba = best_model.predict_proba(X_train)[:, 1]
                y_test_proba = best_model.predict_proba(X_test)[:, 1]
            else:
                y_train_proba = best_model.decision_function(X_train)
                y_test_proba = best_model.decision_function(X_test)

            # calculate metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)

            f1_score_test = f1_score(y_test, y_test_pred, zero_division=0)
            precision_test = precision_score(y_test, y_test_pred, zero_division=0)
            recall_test = recall_score(y_test, y_test_pred, zero_division=0)

            model_report[model_name] = {
                'train_accuracy': train_acc,
                'train_auc': train_auc,
                'test_accuracy': test_acc,
                'test_auc': test_auc,
                'f1_score': f1_score_test,
                'precision': precision_test,
                'recall': recall_test
            }

            models_dict[model_name] = best_model

        return model_report, models_dict
        
    except Exception as e:
        raise CustomException(e, sys)