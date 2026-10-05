import os
import sys

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def evaluate_model(X_train, y_train, models, params):
    try:
        report= {}
        best_models={}
        for model_name, model in models.items():
            par_grid = params[model_name]
            gs_model = GridSearchCV(model, par_grid, cv=5, n_jobs=-1, scoring="accuracy")
            gs_model.fit(X_train, y_train)
            best_model = gs_model.best_estimator_
            cv_score = gs_model.best_score_
            # model.fit(X_train, y_train)
            # acc_train = accuracy_score(y_train, best_model.predict(X_train))
            # acc_test = accuracy_score(y_test, best_model.predict(X_test))
            report[model_name]={"cv_score":cv_score, "best_params": gs_model.best_params_}
            best_models[model_name] = best_model
              
            
        return report, best_models
    except Exception as e:
        print(f"error training the model: {e}")
        raise
        

def save_obj(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        print(f"File save error: {e}")
        raise        

def load_obj(file_path):
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)

        return obj

    except Exception as e:
        print(f"File load error: {e}")
        raise    