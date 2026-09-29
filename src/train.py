import os
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from src.utils import save_obj, evaluate_model

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class ModelTrainConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl") 
    
    
class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainConfig()   
        self.report = {} 
    
    def build_model_trainer(self, train_data, test_data):
        try:
            X_train, y_train, X_test, y_test = (
                train_data.iloc[:,:-1],
                train_data.iloc[:,-1],
                test_data.iloc[:, :-1],
                test_data.iloc[:, -1]
            )
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random forest": RandomForestClassifier(),
                "Gradient Boost": GradientBoostingClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGBOOST": XGBClassifier(),
                "KNN": KNeighborsClassifier()
            }
            
            report = evaluate_model(X_train=X_train, y_train= y_train, X_test=X_test, y_test=y_test, models=models)
            self.report = report
            best_model_name, best_model_info = max(
                report.items(),
                key = lambda item: item[1]["accuracy_test"]
            )
            best_model_test_score = best_model_info["accuracy_test"]
            
            best_model = models[best_model_name]
            
            save_obj(best_model, self.model_train_config.trained_model_path)
            print(f"Best model: {best_model_name}")
            print(f"Best model test score: {best_model_test_score}")
            return best_model
            
        except Exception as e:
            print(f"Error building the model trainer: {e}")
            raise
    def show_report(self):
        for model_name, scores in self.report.items():
            print(f"\nModel: {model_name}")
            print(f"Train Accuracy: {scores['accuracy_train']}")
            print(f"Test Accuracy: {scores['accuracy_test']}")
  
            
        
           
