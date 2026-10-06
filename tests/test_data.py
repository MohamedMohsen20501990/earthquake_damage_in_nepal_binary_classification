import pandas as pd
import pytest
from pydantic import ValidationError

from src.data import DataConfig, DataHandler, TableIdentifier
from src.logging_config import logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_handler():
    logger.info("Creating DataHandler fixture")

    return DataHandler(
        server="localhost",
        driver="ODBC Driver 18 for SQL Server",
        database="TestDatabase",
    )


@pytest.fixture
def sample_dataframe():
    """Sample data with enough rows for stratified splitting."""
    return pd.DataFrame(
        {
            "district_id": [36] * 20,
            "damage_grade": [
                "Grade 1", "Grade 2", "Grade 1", "Grade 2",
                "Grade 3", "Grade 3", "Grade 4", "Grade 4",
                "Grade 5", "Grade 5", "Grade 1", "Grade 2",
                "Grade 1", "Grade 2", "Grade 4", "Grade 5",
                "Grade 1", "Grade 3", "Grade 4", "Grade 5",
            ],
            "post_eq_damage": [0, 1, 0, 1, 0, 1, 1, 1, 1, 1,
                               0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
            "count_floors_pre_eq": [2] * 20,
            "age": list(range(10, 30)),
        }
    )


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

def test_data_config():
    logger.info("Testing DataConfig")

    config = DataConfig()

    assert config.train_data_path == "artifacts/train.csv"
    assert config.test_data_path == "artifacts/test.csv"
    assert config.raw_data_path == "artifacts/data.csv"


# ---------------------------------------------------------------------------
# TableIdentifier
# ---------------------------------------------------------------------------

def test_valid_table_identifier():
    logger.info("Testing valid table identifier")

    table = TableIdentifier(table_name="dbo.earthquake_data")

    assert table.table_name == "dbo.earthquake_data"


@pytest.mark.parametrize(
    "table_name",
    [
        "earthquake_data",
        "dbo.",
        ".earthquake_data",
        "dbo.earthquake-data",
        "dbo.earthquake data",
        "dbo.users;DROP TABLE users",
    ],
)
def test_invalid_table_identifier(table_name):
    logger.info(f"Testing invalid table: {table_name}")

    with pytest.raises(ValidationError):
        TableIdentifier(table_name=table_name)


# ---------------------------------------------------------------------------
# wrangle_data
# ---------------------------------------------------------------------------

def test_wrangle_data(data_handler, sample_dataframe):
    logger.info("Testing wrangle_data")

    original = sample_dataframe.copy()

    result = data_handler.wrangle_data(
        data_frame=sample_dataframe
    )

    # Gorkha filtering
    assert len(result) == 20

    # Feature creation
    assert "severe_damage" in result.columns

    # Grade 1-3 -> 0, Grade 4-5 -> 1
    expected = [
        0, 0, 0, 0,
        0, 0, 1, 1,
        1, 1, 0, 0,
        0, 0, 1, 1,
        0, 0, 1, 1,
    ]

    assert result["severe_damage"].tolist() == expected

    # Removed columns
    for column in [
        "district_id",
        "damage_grade",
        "post_eq_damage",
        "count_floors_pre_eq",
    ]:
        assert column not in result.columns

    # Original DataFrame should remain unchanged
    pd.testing.assert_frame_equal(
        sample_dataframe,
        original,
    )


def test_wrangle_data_input_validation(data_handler, sample_dataframe):
    logger.info("Testing wrangle_data input validation")

    with pytest.raises(
        ValueError,
        match="Etither DataFrame or path must be provided",
    ):
        data_handler.wrangle_data()

    with pytest.raises(
        ValueError,
        match="Privide either a DataFrame or path",
    ):
        data_handler.wrangle_data(
            data_frame=sample_dataframe,
            path="data.csv",
        )


def test_wrangle_data_from_csv(
    data_handler,
    sample_dataframe,
    tmp_path,
):
    logger.info("Testing wrangle_data with CSV")

    csv_path = tmp_path / "data.csv"
    sample_dataframe.to_csv(csv_path, index=False)

    result = data_handler.wrangle_data(
        path=str(csv_path)
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 20
    assert "severe_damage" in result.columns


# ---------------------------------------------------------------------------
# split_and_save
# ---------------------------------------------------------------------------

def test_split_and_save(
    data_handler,
    sample_dataframe,
    tmp_path,
):
    logger.info("Testing split_and_save")

    data_handler.data_config.train_data_path = str(
        tmp_path / "train.csv"
    )
    data_handler.data_config.test_data_path = str(
        tmp_path / "test.csv"
    )
    data_handler.data_config.raw_data_path = str(
        tmp_path / "data.csv"
    )

    clean_df = data_handler.wrangle_data(
        data_frame=sample_dataframe
    )

    data_handler.split_and_save(clean_df)

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    raw_path = tmp_path / "data.csv"

    assert train_path.exists()
    assert test_path.exists()
    assert raw_path.exists()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert len(train_df) + len(test_df) == len(clean_df)
    assert len(train_df) == 16
    assert len(test_df) == 4


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

def test_read_sql_table(data_handler, monkeypatch):
    logger.info("Testing read_sql_table")

    expected = pd.DataFrame({"id": [1, 2]})

    def mock_read_sql(sql, con):
        assert sql == "select * from dbo.earthquake_data"
        assert con == data_handler.engine
        return expected

    monkeypatch.setattr(pd, "read_sql", mock_read_sql)

    result = data_handler.read_sql_table(
        "dbo.earthquake_data"
    )

    pd.testing.assert_frame_equal(result, expected)


def test_read_sql_table_rejects_invalid_name(data_handler):
    logger.info("Testing SQL table validation")

    with pytest.raises(ValidationError):
        data_handler.read_sql_table(
            "users; DROP TABLE users"
        )


def test_save_to_sql(
    data_handler,
    sample_dataframe,
    monkeypatch,
):
    logger.info("Testing save_to_sql")

    captured = {}

    def mock_to_sql(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        pd.DataFrame,
        "to_sql",
        mock_to_sql,
    )

    data_handler.save_to_sql(
        sample_dataframe,
        "dbo.test_table",
    )

    assert captured["name"] == "test_table"
    assert captured["schema"] == "dbo"
    assert captured["con"] == data_handler.engine
    assert captured["if_exists"] == "replace"
    assert captured["index"] is False


def test_save_to_sql_rejects_invalid_name(
    data_handler,
    sample_dataframe,
):
    logger.info("Testing save_to_sql validation")

    with pytest.raises(ValidationError):
        data_handler.save_to_sql(
            sample_dataframe,
            "test_table; DROP TABLE users",
        )