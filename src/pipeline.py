"""Notebook orchestration helpers for the Part D pipeline."""

from collections.abc import Callable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.cleaning import build_silver, run_quality_checks
from src.features import add_next_year_label, build_prescriber_features
from src.ingestion import ingest_year, write_delta
from src.model import run_experiment


def run_bronze(
    spark: SparkSession,
    years: list[int],
    raw_csv_path: Callable[[int], str],
    bronze_table: Callable[[int], str],
    states: list[str] | None,
) -> list[dict]:
    """Load raw CSVs into year-specific bronze Delta tables."""
    return [
        {
            "year": year,
            "table": bronze_table(year),
            "rows": ingest_year(spark, raw_csv_path(year), bronze_table(year), states=states),
        }
        for year in years
    ]


def run_silver(
    spark: SparkSession,
    years: list[int],
    bronze_table: Callable[[int], str],
    silver_table: str,
    dq_table: str,
) -> list[dict]:
    """Build silver rows for each year and append DQ results."""
    summaries = []
    for year in years:
        silver, dq_before = build_silver(spark.table(bronze_table(year)), year)
        write_delta(silver, silver_table, replace_where=f"year = {year}")

        written = spark.table(silver_table).filter(F.col("year") == year)
        _save_dq_results(spark, dq_before + run_quality_checks(written, "after_cleaning"), dq_table, year)
        summaries.append(
            {
                "year": year,
                "bronze_rows": spark.table(bronze_table(year)).count(),
                "silver_rows": written.count(),
            }
        )
    return summaries


def run_gold(spark: SparkSession, years: list[int], silver_table: str, gold_table: str) -> None:
    """Aggregate silver rows into the prescriber-year gold table."""
    for year in years:
        silver = spark.table(silver_table).filter(F.col("year") == year)
        write_delta(build_prescriber_features(silver), gold_table, replace_where=f"year = {year}")


def run_model(
    spark: SparkSession,
    gold_table: str,
    feature_year: int,
    label_year: int,
    states: list[str] | None,
) -> list[dict]:
    """Create labels and run the baseline plus ML models."""
    gold = spark.table(gold_table)
    training_data = add_next_year_label(
        gold.filter(F.col("year") == feature_year),
        gold.filter(F.col("year") == label_year),
    )
    return run_experiment(
        training_data,
        {
            "feature_year": feature_year,
            "label_year": label_year,
            "states": ",".join(states) if states else "ALL",
        },
    )


def _save_dq_results(spark: SparkSession, results: list[dict], table_name: str, year: int) -> None:
    dq = (
        spark.createDataFrame(results)
        .withColumn("year", F.lit(year))
        .withColumn("checked_at", F.current_timestamp())
    )
    dq.write.format("delta").mode("append").saveAsTable(table_name)
