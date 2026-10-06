import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.utils import evaluate_model, save_obj, load_obj
from src.logging_config import logger


def test_evaluate_model():
    logger.info("Testing evaluate_model")

    X_train = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature_2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    })
    y_train = pd.Series([0, 1] * 5)

    models = {
        "DummyModel": DummyClassifier(strategy="most_frequent")
    }

    params = {
        "DummyModel": {
            "strategy": ["most_frequent", "prior"]
        }
    }

    report, best_models = evaluate_model(
        X_train=X_train,
        y_train=y_train,
        models=models,
        params=params,
    )

    assert "DummyModel" in report
    assert "cv_score" in report["DummyModel"]
    assert "best_params" in report["DummyModel"]

    assert "DummyModel" in best_models
    assert hasattr(best_models["DummyModel"], "predict")

    logger.info("evaluate_model test passed")


def test_evaluate_model_invalid_params():
    logger.info("Testing evaluate_model with invalid parameters")

    X_train = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5],
    })
    y_train = pd.Series([0, 1, 0, 1, 0])

    models = {
        "DummyModel": DummyClassifier()
    }

    params = {
        "DummyModel": {
            "invalid_parameter": [1, 2]
        }
    }

    with pytest.raises(ValueError):
        evaluate_model(X_train, y_train, models, params)

    logger.info("Invalid parameters correctly rejected")


def test_save_and_load_obj(tmp_path):
    logger.info("Testing save_obj and load_obj")

    file_path = tmp_path / "test.pkl"

    obj = {
        "name": "earthquake",
        "accuracy": 0.95,
        "values": [1, 2, 3],
    }

    save_obj(obj, str(file_path))

    assert file_path.exists()

    loaded_obj = load_obj(str(file_path))

    assert loaded_obj == obj

    logger.info("save_obj and load_obj tests passed")


def test_save_obj_creates_directory(tmp_path):
    logger.info("Testing save_obj directory creation")

    file_path = tmp_path / "artifacts" / "model.pkl"

    obj = {"model": "test"}

    save_obj(obj, str(file_path))

    assert file_path.exists()
    assert load_obj(str(file_path)) == obj

    logger.info("Directory creation test passed")


def test_load_obj_invalid_path(tmp_path):
    logger.info("Testing load_obj with invalid path")

    file_path = tmp_path / "does_not_exist.pkl"

    with pytest.raises(FileNotFoundError):
        load_obj(str(file_path))

    logger.info("Invalid file path correctly rejected")


def test_save_obj_invalid_path():
    logger.info("Testing save_obj with invalid path")

    with pytest.raises((OSError, IOError)):
        save_obj({"test": "data"}, "/invalid/path/model.pkl")

    logger.info("Invalid save path correctly rejected")
