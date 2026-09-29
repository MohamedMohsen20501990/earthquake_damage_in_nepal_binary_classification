import os
import sys

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report= {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            acc_train = accuracy_score(y_train, model.predict(X_train))
            acc_test = accuracy_score(y_test, model.predict(X_test))
            report[model_name]={"accuracy_train": acc_train, "accuracy_test": acc_test}
        return report
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