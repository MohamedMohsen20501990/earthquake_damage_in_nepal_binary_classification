import os
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
from src.utils import save_obj



import warnings
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class PreprocessorConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    transformed_train_paths: str = os.path.join("artifacts", "train_transformed.parquet")
    transformed_test_path: str = os.path.join("artifacts", "test_transformed.parquet")


class Preprocessor:
    def __init__(self):
        self.preprocessor_config = PreprocessorConfig()
        
    
    def build_preprocessor(self):
        try:
            cat_cols = ['land_surface_condition', 'foundation_type', 'roof_type',
                        'ground_floor_type', 'other_floor_type', 'position',
                        'plan_configuration']
            
            num_cols = ['age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq'] 
            
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="mean")),
                ('scaler', RobustScaler())
            ])
            
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('encoder', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', numerical_pipeline, num_cols),
                ('cat_pipeline', categorical_pipeline, cat_cols)
            ])
            
            return preprocessor
            
        except Exception as e:
            print(f"error: preprocessor building {e}")
            raise

    
    def start_data_preprocessing(self, train_path, test_path):
        try:
            train_set = pd.read_csv(train_path)
            test_set = pd.read_csv(test_path)
            target="severe_damage"
            
            X_train = train_set.drop(columns=[target])
            y_train = train_set[target]
            
            
            X_test = test_set.drop(columns=[target])
            y_test = test_set[target]
            
            preprocessor_obj = self.build_preprocessor()
            
            transformed_train_data = preprocessor_obj.fit_transform(X_train)
  
            
            transformed_test_data = preprocessor_obj.transform(X_test)

            
            feature_names = preprocessor_obj.get_feature_names_out()
            
            train_transformed_df = pd.DataFrame(transformed_train_data, columns=feature_names)
            train_transformed_df[target] = y_train.values
            
            test_transformed_df = pd.DataFrame(transformed_test_data, columns=feature_names)
            test_transformed_df[target] = y_test.values
            
            # saving the preprocessor
            save_obj(preprocessor_obj, self.preprocessor_config.preprocessor_path)
            
            # saving the transformed datasets to parquet format
            train_transformed_df.to_parquet(self.preprocessor_config.transformed_train_path, index=False)
            test_transformed_df.to_parquet(self.preprocessor_config.transformed_test_path, index=False)
            return (
                train_transformed_df,
                test_transformed_df, 
                preprocessor_obj
            )
        except Exception as e:
            print(f"error starting preprocessing: {e}")
            raise