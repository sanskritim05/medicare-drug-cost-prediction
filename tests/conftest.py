import os
import sys

import pytest
from pyspark.sql import SparkSession

from src.cleaning import COLUMN_MAP


@pytest.fixture(scope="session")
def spark():
    """A small local Spark session shared by all tests."""
    # Make Spark's worker processes use the same Python as the test runner.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    session = (
        SparkSession.builder.master("local[1]")
        .appName("partd-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.ansi.enabled", "true")  # match Databricks serverless
        .getOrCreate()
    )
    yield session
    session.stop()


def raw_row(**overrides) -> dict:
    """One raw CMS row (all strings, raw column names) with sensible defaults."""
    row = {
        "Prscrbr_NPI": "1234567890",
        "Prscrbr_Last_Org_Name": "Smith",
        "Prscrbr_First_Name": "Ann",
        "Prscrbr_City": "Providence",
        "Prscrbr_State_Abrvtn": "RI",
        "Prscrbr_Type": "Internal Medicine",
        "Brnd_Name": "Lipitor",
        "Gnrc_Name": "Atorvastatin Calcium",
        "Tot_Clms": "20",
        "Tot_30day_Fills": "25.5",
        "Tot_Day_Suply": "750",
        "Tot_Drug_Cst": "1000.00",
        "Tot_Benes": "12",
        # An extra raw column that silver should drop.
        "GE65_Tot_Clms": "15",
    }
    row.update(overrides)
    return row


@pytest.fixture
def make_raw_df(spark):
    """Build a raw (bronze-shaped) DataFrame from a list of row dicts."""

    def _make(rows: list[dict]):
        columns = list(COLUMN_MAP) + ["GE65_Tot_Clms"]
        return spark.createDataFrame([[r.get(c) for c in columns] for r in rows], columns)

    return _make
