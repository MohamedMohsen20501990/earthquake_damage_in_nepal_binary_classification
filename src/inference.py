import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils import load_obj
from src.logging_config import logger




class InferencePipeline:
    """
    Handles the model inference process by loading the trained model
    and preprocessor, transforming input data, and generating predictions.
    """
    def __init__(self):
        logger.info("Initializing InferencePipeline")
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"
        logger.debug(f"Loading model from: {self.model_path}")
        self.model = load_obj(self.model_path)
        logger.info("Model loaded successfully")
        logger.debug(f"Loading preprocessor from: {self.preprocessor_path}")
        self.preprocessor = load_obj(self.preprocessor_path)
        logger.info("Preprocessor loaded successfully")
        logger.info("InferencePipeline initialized successfully")
    
    def predict(self, input_data):
        """
        Transform input data and generate a prediction.

        Parameters
        ----------
        input_data : InputData
            Input building features.

        Returns
        -------
        dict
            Building ID and predicted damage class.
        """
        try:
            logger.info("Starting prediction")
            logger.debug(
                f"Converting input data to DataFrame | "
                f"building_id={input_data.building_id}"
            )
            df = input_data.input_data_to_data_frame()
            building_id = input_data.building_id
            features = df.drop(columns=["building_id"])
            transformed_data = self.preprocessor.transform(features)
            prediction = self.model.predict(transformed_data)
            return {"building_id": building_id,  "prediction": int(prediction[0])}
        except Exception:
            logger.exception("Error while prediction")
            raise


        
    
    
class InputData:
    """
    Stores building features provided as input for model inference.
    """
    def __init__(
        self,
        building_id: int,
        age_building: int,
        plinth_area_sq_ft: float,
        height_ft_pre_eq: float,
        land_surface_condition: str,
        foundation_type: str,
        roof_type: str,
        ground_floor_type: str,
        other_floor_type: str,
        position: str,
        plan_configuration: str
        
    ):
        self.building_id = building_id
        self.age_building = age_building
        self.plinth_area_sq_ft = plinth_area_sq_ft
        self.height_ft_pre_eq = height_ft_pre_eq
        self.land_surface_condition = land_surface_condition
        self.foundation_type = foundation_type
        self.roof_type = roof_type
        self.ground_floor_type = ground_floor_type
        self.other_floor_type = other_floor_type
        self.position = position
        self.plan_configuration = plan_configuration
        
    def input_data_to_data_frame(self):
        """
        Convert the input attributes into a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            Single-row DataFrame containing the input features.
        """
        try:
            input_data = {
                "building_id": [self.building_id],
                "age_building": [self.age_building],
                "plinth_area_sq_ft": [self.plinth_area_sq_ft],
                "height_ft_pre_eq": [self.height_ft_pre_eq],
                "land_surface_condition": [self.land_surface_condition],
                "foundation_type": [self.foundation_type],
                "roof_type": [self.roof_type],
                "ground_floor_type": [self.ground_floor_type],
                "other_floor_type": [self.other_floor_type],
                "position": [self.position],
                "plan_configuration": [self.plan_configuration]
            }
            df = pd.DataFrame(input_data)
            return df
        except Exception:
            logger.exception("Error converting input data to DataFrame")

if __name__ == "__main__":
    input_data = InputData(
        building_id=123124543,
        age_building=32,
        plinth_area_sq_ft=285,
        height_ft_pre_eq=9,
        land_surface_condition="flat",
        foundation_type="RC",
        roof_type="Bamboo/Timber-Light roof	Mud",
        ground_floor_type="Mud",
        other_floor_type="Not applicable",
        position="Not attached",
        plan_configuration="Rectangular"
    )
    inference_pipeline = InferencePipeline()
    result = inference_pipeline.predict(input_data)
    print(result)
    