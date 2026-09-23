import os
import sys

import pandas as pd
import numpy as np
import pickle

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

# def load_obj(file_path):
#     try:
#         with open(file_path, "rb") as f:
#             obj = pickle.load(f)

#         return obj

#     except Exception as e:
#         print(f"File load error: {e}")
#         raise    