import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.train import ModelTrainConfig, ModelTrainer
from src.logging_config import logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_trainer():
    logger.info("Creating ModelTrainer fixture")
    return ModelTrainer()


@pytest.fixture
def sample_train_test_data():
    train_data = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5, 6],
            "feature_2": [10, 20, 30, 40, 50, 60],
            "severe_damage": [0, 0, 0, 1, 1, 1],
        }
    )

    test_data = pd.DataFrame(
        {
            "feature_1": [7, 8, 9, 10],
            "feature_2": [70, 80, 90, 100],
            "severe_damage": [0, 1, 1, 1],
        }
    )

    return train_data, test_data


def fitted_dummy_models(train_data):
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    model_a = DummyClassifier(strategy="most_frequent")
    model_b = DummyClassifier(strategy="prior")

    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    return model_a, model_b


# ---------------------------------------------------------------------------
# Config & initialization
# ---------------------------------------------------------------------------

def test_model_train_config():
    logger.info("Testing ModelTrainConfig")

    config = ModelTrainConfig()

    assert config.trained_model_path == "artifacts/model.pkl"


def test_model_trainer_initialization(model_trainer):
    logger.info("Testing ModelTrainer initialization")

    assert model_trainer.report == {}
    assert model_trainer.best_model_name is None
    assert model_trainer.test_accuracy is None


# ---------------------------------------------------------------------------
# build_model_trainer
# ---------------------------------------------------------------------------

def test_build_model_trainer(
    model_trainer,
    sample_train_test_data,
    monkeypatch,
    tmp_path,
):
    logger.info("Testing build_model_trainer")

    train_data, test_data = sample_train_test_data

    model_a, model_b = fitted_dummy_models(train_data)

    fake_report = {
        "DummyModel": {
            "cv_score": 0.90,
            "best_params": {},
        },
        "OtherModel": {
            "cv_score": 0.80,
            "best_params": {},
        },
    }

    fake_best_models = {
        "DummyModel": model_a,
        "OtherModel": model_b,
    }

    def mock_evaluate_model(
        X_train,
        y_train,
        models,
        params,
    ):
        logger.info("Mock evaluate_model called")

        assert len(X_train) == len(train_data)
        assert len(y_train) == len(train_data)

        assert "LogisticRegression" in models
        assert "RandomForest" in models
        assert "GradientBoost" in models
        assert "DecisionTree" in models
        assert "XGBOOST" in models
        assert "KNN" in models

        return fake_report, fake_best_models

    saved = {}

    def mock_save_obj(obj, path):
        logger.info("Mock save_obj called")
        saved["model"] = obj
        saved["path"] = path

    monkeypatch.setattr(
        "src.train.evaluate_model",
        mock_evaluate_model,
    )

    monkeypatch.setattr(
        "src.train.save_obj",
        mock_save_obj,
    )

    model_path = tmp_path / "model.pkl"

    model_trainer.model_train_config.trained_model_path = str(
        model_path
    )

    result = model_trainer.build_model_trainer(
        train_data=train_data,
        test_data=test_data,
    )

    assert result is model_a
    assert model_trainer.best_model_name == "DummyModel"
    assert model_trainer.test_accuracy == 0.25
    assert model_trainer.report == fake_report

    assert saved["model"] is model_a
    assert saved["path"] == str(model_path)


# ---------------------------------------------------------------------------
# Best model selection
# ---------------------------------------------------------------------------

def test_best_model_selected_by_cv_score(
    model_trainer,
    sample_train_test_data,
    monkeypatch,
):
    logger.info("Testing best model selection")

    train_data, test_data = sample_train_test_data

    model_a, model_b = fitted_dummy_models(train_data)

    fake_report = {
        "ModelA": {
            "cv_score": 0.70,
            "best_params": {},
        },
        "ModelB": {
            "cv_score": 0.95,
            "best_params": {},
        },
    }

    fake_models = {
        "ModelA": model_a,
        "ModelB": model_b,
    }

    def mock_evaluate_model(*args, **kwargs):
        return fake_report, fake_models

    monkeypatch.setattr(
        "src.train.evaluate_model",
        mock_evaluate_model,
    )

    monkeypatch.setattr(
        "src.train.save_obj",
        lambda *args, **kwargs: None,
    )

    model_trainer.build_model_trainer(
        train_data=train_data,
        test_data=test_data,
    )

    assert model_trainer.best_model_name == "ModelB"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_build_model_trainer_raises_error(
    model_trainer,
    sample_train_test_data,
    monkeypatch,
):
    logger.info("Testing build_model_trainer error handling")

    train_data, test_data = sample_train_test_data

    def mock_evaluate_model(*args, **kwargs):
        raise RuntimeError("Evaluation failed")

    monkeypatch.setattr(
        "src.train.evaluate_model",
        mock_evaluate_model,
    )

    with pytest.raises(
        RuntimeError,
        match="Evaluation failed",
    ):
        model_trainer.build_model_trainer(
            train_data=train_data,
            test_data=test_data,
        )


# ---------------------------------------------------------------------------
# show_report
# ---------------------------------------------------------------------------

def test_show_report(model_trainer, capsys):
    logger.info("Testing show_report")

    model_trainer.report = {
        "LogisticRegression": {
            "cv_score": 0.90,
            "best_params": {
                "max_iter": 1000,
            },
        }
    }

    model_trainer.best_model_name = "LogisticRegression"
    model_trainer.test_accuracy = 0.88

    model_trainer.show_report()

    output = capsys.readouterr().out

    assert "LogisticRegression" in output
    assert "CV Score: 0.9" in output
    assert "Best Model: LogisticRegression" in output
    assert "Test Accuracy: 0.88" in output