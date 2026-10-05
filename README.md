# Earthquake in Nepal: Building Damage Prediction


Data

Download the dataset from:
<[dataset link](https://www.kaggle.com/datasets/domingosnhamussusa/earthquake-damage-in-nepal)>

Place it in:

data/raw/earthquake_data.csv

## Project Overview

This project focuses on a **binary classification problem** to predict the likelihood of a building suffering **severe damage due to an earthquake in Nepal**.

Using data from the **2015 Nepal (Gorkha) earthquake**, the goal is to analyze structural and environmental features of buildings and build a machine learning model that can estimate damage risk.

---

## Objective

To develop a machine learning model that predicts whether a building is likely to experience **severe structural damage** during an earthquake.

This can support:
- Disaster risk assessment  
- Emergency planning  
- Safer infrastructure design  

---

## Project Structure
```
├── README.md <- Top-level project documentation
│
├── data
│ ├── raw <- Original, immutable dataset
│ └── processed <- Cleaned and feature-engineered data
│
├── models <- Trained models and serialized outputs
│
├── notebooks <- Jupyter notebooks for EDA and experimentation
│
├── reports <- Generated analysis reports (HTML, PDF, etc.)
│ └── figures <- Visualizations and plots used in reports
│
├── requirements.txt <- Python dependencies for reproducibility
│
└── src <- Source code (data processing, training, evaluation, utility modules)
---```

## Workflow

1. Data collection and exploration (EDA)
2. Data cleaning and preprocessing
3. Feature engineering and selection
4. Model training (binary classification)
5. Model evaluation and validation
6. Communicating results

---
classDiagram

    %% =========================
    %% Data Processing
    %% =========================

    class DataConfig {
        +str train_data_path
        +str test_data_path
        +str raw_data_path
    }

    class TableIdentifier {
        +str table_name
        +validate_table_name(value) str
    }

    class DataHandler {
        +DataConfig data_config
        +str server
        +str driver
        +str database
        +engine engine
        +read_sql_table(table_name) DataFrame
        +wrangle_data(data_frame, path, index_col) DataFrame
        +save_to_sql(data_frame, table_name)
        +split_and_save(data_frame)
    }

    %% =========================
    %% Preprocessing
    %% =========================

    class PreprocessorConfig {
        +str preprocessor_path
        +str transformed_train_path
        +str transformed_test_path
    }

    class Preprocessor {
        +PreprocessorConfig preprocessor_config
        +build_preprocessor() ColumnTransformer
        +start_data_preprocessing(train_path, test_path)
    }

    %% =========================
    %% Model Training
    %% =========================

    class ModelTrainConfig {
        +str trained_model_path
    }

    class ModelTrainer {
        +ModelTrainConfig model_train_config
        +dict report
        +str best_model_name
        +float test_accuracy
        +build_model_trainer(train_data, test_data)
        +show_report()
    }

    %% =========================
    %% Inference
    %% =========================

    class InputData {
        +int building_id
        +int age_building
        +float plinth_area_sq_ft
        +float height_ft_pre_eq
        +str land_surface_condition
        +str foundation_type
        +str roof_type
        +str ground_floor_type
        +str other_floor_type
        +str position
        +str plan_configuration
        +input_data_to_data_frame() DataFrame
    }

    class InferencePipeline {
        +str model_path
        +str preprocessor_path
        +model model
        +preprocessor preprocessor
        +predict(input_data) dict
    }

    %% =========================
    %% API Layer
    %% =========================

    class DataIN {
        +int building_id
        +int age_building
        +float plinth_area_sq_ft
        +float height_ft_pre_eq
        +str land_surface_condition
        +str foundation_type
        +str roof_type
        +str ground_floor_type
        +str other_floor_type
        +str position
        +str plan_configuration
    }

    class DataOut {
        +int building_id
        +bool success
        +int prediction
        +str message
    }

    %% =========================
    %% Relationships
    %% =========================

    DataHandler --> DataConfig : uses
    DataHandler --> TableIdentifier : validates

    DataHandler --> Preprocessor : provides train/test data
    Preprocessor --> PreprocessorConfig : uses

    Preprocessor --> ModelTrainer : provides transformed data
    ModelTrainer --> ModelTrainConfig : uses

    InputData --> InferencePipeline : input
    InferencePipeline --> Preprocessor : transforms features
    InferencePipeline --> ModelTrainer : loads trained model

    DataIN --> InputData : creates
    DataIN --> InferencePipeline : request

    InferencePipeline --> DataOut : prediction


## Dataset

The Nepal Earthquake Damage Dataset contains information about buildings affected by the 2015 Nepal earthquake.

Source:
https://eq2015.npc.gov.np/#/download

Features include building age, height, foundation type, roof type, land surface condition, and other structural characteristics.

Target:
- damage_grade

## Expected Outcome

A trained classification model capable of predicting whether a building will suffer **severe damage**, helping improve earthquake preparedness and risk mitigation strategies.

---
