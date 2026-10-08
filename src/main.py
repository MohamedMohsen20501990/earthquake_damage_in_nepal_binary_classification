import os
import sys
from pathlib import Path
# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from src.inference import InputData, InferencePipeline
from pydantic import BaseModel, Field
from typing import Literal
from src.logging_config import logger

class DataIN(BaseModel):
    building_id: int = Field(gt=0)
    age_building: int = Field (gt=0, le=200)
    plinth_area_sq_ft: float = Field(gt=0)
    height_ft_pre_eq: float = Field(ge=0)
    land_surface_condition: Literal['Flat', 'Moderate slope', 'Steep slope']
    foundation_type: Literal['Other', 'Mud mortar-Stone/Brick', 'Cement-Stone/Brick', 'Bamboo/Timber','RC']
    roof_type: Literal['Bamboo/Timber-Light roof', 'Bamboo/Timber-Heavy roof', 'RCC/RB/RBC']
    ground_floor_type: Literal['Mud', 'Brick/Stone', 'RC', 'Timber', 'Other']
    other_floor_type: Literal['Not applicable', 'TImber/Bamboo-Mud', 'Timber-Planck', 'RCC/RB/RBC']
    position: Literal['Not attached', 'Attached-1 side', 'Attached-2 side', 'Attached-3 side'] 
    plan_configuration: Literal['Rectangular','L-shape','Square','T-shape','Multi-projected','H-shape','U-shape','Others','E-shape', 'Building with Central Courtyard']


class DataOut(BaseModel):
    building_id:int
    success: bool
    prediction : int | None = None
    message: str = Field(min_length=1, max_length=200)
    
    
app = FastAPI()  
    
@app.post("/predict", status_code=200, response_model=DataOut)  
def predict(request: DataIN):
    logger.info("Prediction endpoint started")
    logger.debug(f"Received prediction request for building_id={request.building_id}")
    response = request.model_dump()
    
    try:
        logger.info("Creating InputData object")
        input_data = InputData(
            building_id= request.building_id,
            age_building= request.age_building,
            plinth_area_sq_ft= request.plinth_area_sq_ft,
            height_ft_pre_eq= request.height_ft_pre_eq,
            land_surface_condition=request.land_surface_condition,
            foundation_type= request.foundation_type,
            roof_type=request.roof_type,
            ground_floor_type=request.ground_floor_type,
            other_floor_type= request.other_floor_type,
            position=request.position,
            plan_configuration=request.plan_configuration
        )
        logger.info("InputData object created successfully")
        logger.info("Initializing InferencePipeline")
        
        inference_pipeline = InferencePipeline()
        prediction = inference_pipeline.predict(input_data=input_data)
        response["success"] = True
        response["prediction"] = prediction["prediction"]
        response["message"] = "Prediction Done"
        
    except Exception as e:
        response["success"] = False
        response["prediction"] = None
        response["message"] = str(e)
    return response   

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )



