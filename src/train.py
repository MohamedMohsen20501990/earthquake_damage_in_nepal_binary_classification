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
        self.best_model_name=None
        self.test_accuracy=None
    
    def build_model_trainer(self, train_data, test_data):
        try:
            X_train, y_train, X_test, y_test = (
                train_data.iloc[:,:-1],
                train_data.iloc[:,-1],
                test_data.iloc[:, :-1],
                test_data.iloc[:, -1]
            )
            
            models = {
                "LogisticRegression": LogisticRegression(solver="liblinear"),
                "RandomForest": RandomForestClassifier(),
                "GradientBoost": GradientBoostingClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "XGBOOST": XGBClassifier(),
                "KNN": KNeighborsClassifier()
            }
            
            params = {
                "LogisticRegression":{
                    "max_iter":[100,1000]
                },
                "RandomForest":{
                    "n_estimators":[50,100,150],
                    "max_depth":[5,10,15]
                },
                "GradientBoost":{
                    "max_depth":[5,10,15],
                    "n_estimators":[50,100,151]
                },
                "DecisionTree":{
                    "max_depth":[5,10,15]
                },
                "XGBOOST":{
                    "max_depth":range(5,31,5),
                    "n_estimators":[50,100,151]
                },
                "KNN":{
                    "n_neighbors":[5,10]
                }
                
            }
            
            report, best_models = evaluate_model(X_train=X_train, y_train= y_train, models=models, params=params)
            self.report = report
            best_model_name, best_model_info = max(
                report.items(),
                key = lambda item: item[1]["cv_score"]
            )
            
            best_model = best_models[best_model_name]
            
            test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
            self.best_model_name = best_model_name
            self.test_accuracy = test_accuracy

            
            save_obj(best_model, self.model_train_config.trained_model_path)
            print(f"Best model: {best_model_name}")
            print(f"CV score: {best_model_info['cv_score']}")
            print(f"Test accuracy: {test_accuracy}")
            return best_model
            
        except Exception as e:
            print(f"Error building the model trainer: {e}")
            raise
    def show_report(self):
        for model_name, scores in self.report.items():
            print(f"\nModel: {model_name}")
            print(f"CV Score: {scores['cv_score']}")
            print(f"Best Parameters: {scores['best_params']}")

        print("\n==============================")
        print(f"Best Model: {self.best_model_name}")
        print(f"Test Accuracy: {self.test_accuracy}")




  
            
        
           
