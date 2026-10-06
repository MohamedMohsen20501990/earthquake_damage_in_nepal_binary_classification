import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.preprocess import Preprocessor
from src.logging_config import logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def preprocessor():
    logger.info("Creating Preprocessor fixture")
    return Preprocessor()


@pytest.fixture
def sample_data():
    """Small dataset containing numerical and categorical features."""

    data = pd.DataFrame(
        {
            "age_building": [10, 20, None, 40, 50, 60],
            "plinth_area_sq_ft": [1000, 1200, 1500, None, 1800, 2000],
            "height_ft_pre_eq": [10, 15, 20, 25, None, 35],

            "land_surface_condition": [
                "Flat", "Flat", "Moderate",
                "Flat", "Steep", "Steep"
            ],
            "foundation_type": [
                "Mud", "RC", "Mud",
                "RC", "Mud", "RC"
            ],
            "roof_type": [
                "Bamboo", "RCC", "Bamboo",
                "RCC", "Bamboo", "RCC"
            ],
            "ground_floor_type": [
                "Mud", "Brick", "Mud",
                "Brick", "Mud", "Brick"
            ],
            "other_floor_type": [
                "TImber", "RCC", "TImber",
                "RCC", "TImber", "RCC"
            ],
            "position": [
                "Not attached", "Attached 1 side", "Not attached",
                "Attached 2 sides", "Not attached", "Attached 1 side"
            ],
            "plan_configuration": [
                "Rectangular", "Rectangular", "L-shape",
                "Rectangular", "L-shape", "Rectangular"
            ],

            "severe_damage": [0, 1, 0, 1, 0, 1],
        }
    )

    return data


# ---------------------------------------------------------------------------
# build_preprocessor
# ---------------------------------------------------------------------------

def test_build_preprocessor(preprocessor):
    logger.info("Testing build_preprocessor")

    processor = preprocessor.build_preprocessor()

    assert isinstance(processor, ColumnTransformer)

    transformer_names = [
        name for name, _, _ in processor.transformers
    ]

    assert "num_pipeline" in transformer_names
    assert "cat_pipeline" in transformer_names

    logger.info("build_preprocessor test passed")


def test_preprocessor_pipelines(preprocessor):
    logger.info("Testing numerical and categorical pipelines")

    processor = preprocessor.build_preprocessor()

    transformers = dict(
        (name, transformer)
        for name, transformer, _ in processor.transformers
    )

    numerical_pipeline = transformers["num_pipeline"]
    categorical_pipeline = transformers["cat_pipeline"]

    assert isinstance(numerical_pipeline, Pipeline)
    assert isinstance(categorical_pipeline, Pipeline)

    assert "imputer" in numerical_pipeline.named_steps
    assert "scaler" in numerical_pipeline.named_steps

    assert "imputer" in categorical_pipeline.named_steps
    assert "encoder" in categorical_pipeline.named_steps

    logger.info("Pipeline structure test passed")


# ---------------------------------------------------------------------------
# start_data_preprocessing
# ---------------------------------------------------------------------------

def test_start_data_preprocessing(
    preprocessor,
    sample_data,
    tmp_path,
):
    logger.info("Testing start_data_preprocessing")

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    sample_data.iloc[:4].to_csv(
        train_path,
        index=False,
    )

    sample_data.iloc[4:].to_csv(
        test_path,
        index=False,
    )

    # Redirect output files to pytest temporary directory
    preprocessor.preprocessor_config.preprocessor_path = str(
        tmp_path / "preprocessor.pkl"
    )

    preprocessor.preprocessor_config.transformed_train_path = str(
        tmp_path / "train_transformed.parquet"
    )

    preprocessor.preprocessor_config.transformed_test_path = str(
        tmp_path / "test_transformed.parquet"
    )

    train_result, test_result, processor_path = (
        preprocessor.start_data_preprocessing(
            train_path=str(train_path),
            test_path=str(test_path),
        )
    )

    # Returned objects
    assert isinstance(train_result, pd.DataFrame)
    assert isinstance(test_result, pd.DataFrame)

    # Target must remain
    assert "severe_damage" in train_result.columns
    assert "severe_damage" in test_result.columns

    # Number of rows must remain unchanged
    assert len(train_result) == 4
    assert len(test_result) == 2

    # Preprocessor path returned
    assert processor_path == str(
        tmp_path / "preprocessor.pkl"
    )

    # Saved files should exist
    assert (tmp_path / "preprocessor.pkl").exists()
    assert (tmp_path / "train_transformed.parquet").exists()
    assert (tmp_path / "test_transformed.parquet").exists()

    logger.info("start_data_preprocessing test passed")


def test_preprocessing_handles_missing_values(
    preprocessor,
    sample_data,
    tmp_path,
):
    logger.info("Testing missing value handling")

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    sample_data.iloc[:4].to_csv(
        train_path,
        index=False,
    )

    sample_data.iloc[4:].to_csv(
        test_path,
        index=False,
    )

    preprocessor.preprocessor_config.preprocessor_path = str(
        tmp_path / "preprocessor.pkl"
    )

    preprocessor.preprocessor_config.transformed_train_path = str(
        tmp_path / "train.parquet"
    )

    preprocessor.preprocessor_config.transformed_test_path = str(
        tmp_path / "test.parquet"
    )

    train_result, test_result, _ = (
        preprocessor.start_data_preprocessing(
            train_path=str(train_path),
            test_path=str(test_path),
        )
    )

    # After preprocessing there should be no NaN values.
    assert not train_result.isnull().any().any()
    assert not test_result.isnull().any().any()

    logger.info("Missing value handling test passed")


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

def test_start_data_preprocessing_invalid_path(preprocessor):
    logger.info("Testing invalid input path")

    with pytest.raises(FileNotFoundError):
        preprocessor.start_data_preprocessing(
            train_path="does_not_exist_train.csv",
            test_path="does_not_exist_test.csv",
        )

    logger.info("Invalid path correctly rejected")
