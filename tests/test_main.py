import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.logging_config import logger


client = TestClient(app)


@pytest.fixture
def valid_payload():
    logger.info("Creating valid prediction payload")

    return {
        "building_id": 1,
        "age_building": 20,
        "plinth_area_sq_ft": 1000,
        "height_ft_pre_eq": 15,
        "land_surface_condition": "Flat",
        "foundation_type": "RC",
        "roof_type": "RCC/RB/RBC",
        "ground_floor_type": "RC",
        "other_floor_type": "Not applicable",
        "position": "Not attached",
        "plan_configuration": "Rectangular",
    }


def test_predict_success(valid_payload, monkeypatch):
    logger.info("Testing successful prediction")

    class MockPipeline:
        def predict(self, input_data):
            return {"prediction": 1}

    monkeypatch.setattr(
        "src.main.InferencePipeline",
        MockPipeline,
    )

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200

    data = response.json()

    assert data["building_id"] == 1
    assert data["success"] is True
    assert data["prediction"] == 1
    assert data["message"] == "Prediction Done"

    logger.info("Successful prediction test passed")


def test_predict_returns_zero(valid_payload, monkeypatch):
    logger.info("Testing zero prediction")

    class MockPipeline:
        def predict(self, input_data):
            return {"prediction": 0}

    monkeypatch.setattr(
        "src.main.InferencePipeline",
        MockPipeline,
    )

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == 0

    logger.info("Zero prediction test passed")


def test_predict_handles_pipeline_error(valid_payload, monkeypatch):
    logger.info("Testing inference pipeline error handling")

    class MockPipeline:
        def predict(self, input_data):
            raise Exception("Prediction failed")

    monkeypatch.setattr(
        "src.main.InferencePipeline",
        MockPipeline,
    )

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is False
    assert data["prediction"] is None
    assert data["message"] == "Prediction failed"

    logger.info("Pipeline error handled correctly")


@pytest.mark.parametrize(
    "field,value",
    [
        ("building_id", 0),
        ("age_building", 0),
        ("age_building", 201),
        ("plinth_area_sq_ft", 0),
        ("height_ft_pre_eq", -1),
        ("land_surface_condition", "Unknown"),
        ("foundation_type", "Unknown"),
        ("roof_type", "Unknown"),
        ("ground_floor_type", "Unknown"),
        ("other_floor_type", "Unknown"),
        ("position", "Unknown"),
        ("plan_configuration", "Unknown"),
    ],
)
def test_predict_validation_error(valid_payload, field, value):
    logger.info(
        "Testing validation error for %s=%s",
        field,
        value,
    )

    payload = valid_payload.copy()
    payload[field] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

    logger.info("Validation error correctly returned")


def test_predict_missing_required_field(valid_payload):
    logger.info("Testing missing required field")

    payload = valid_payload.copy()
    del payload["building_id"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

    logger.info("Missing field correctly rejected")


def test_predict_response_structure(valid_payload, monkeypatch):
    logger.info("Testing prediction response structure")

    class MockPipeline:
        def predict(self, input_data):
            return {"prediction": 1}

    monkeypatch.setattr(
        "src.main.InferencePipeline",
        MockPipeline,
    )

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200

    data = response.json()

    assert set(data.keys()) == {
        "building_id",
        "success",
        "prediction",
        "message",
    }

    logger.info("Response structure test passed")
