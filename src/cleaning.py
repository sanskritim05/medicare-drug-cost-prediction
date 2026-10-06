"""Silver layer: rename, type, standardize, validate and de-duplicate.

Pipeline order (see `build_silver`):
    standardize_columns -> cast_types -> standardize_text
    -> run_quality_checks (logged) -> drop_invalid_rows -> drop exact duplicates
"""

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

# Raw CMS column name -> clean snake_case name. Columns not listed are dropped.
COLUMN_MAP = {
    "Prscrbr_NPI": "npi",
    "Prscrbr_Last_Org_Name": "last_org_name",
    "Prscrbr_First_Name": "first_name",
    "Prscrbr_City": "city",
    "Prscrbr_State_Abrvtn": "state",
    "Prscrbr_Type": "specialty",
    "Brnd_Name": "brand_name",
    "Gnrc_Name": "generic_name",
    "Tot_Clms": "total_claims",
    "Tot_30day_Fills": "total_30day_fills",
    "Tot_Day_Suply": "total_day_supply",
    "Tot_Drug_Cst": "total_drug_cost",
    "Tot_Benes": "total_beneficiaries",
}

INTEGER_COLUMNS = ["total_claims", "total_day_supply", "total_beneficiaries"]
DOUBLE_COLUMNS = ["total_30day_fills", "total_drug_cost"]
TEXT_COLUMNS = ["last_org_name", "first_name", "city", "state", "specialty"]
DRUG_NAME_COLUMNS = ["brand_name", "generic_name"]

# One row per prescriber x drug in the CMS file.
KEY_COLUMNS = ["npi", "brand_name", "generic_name"]

# Data quality thresholds: a check fails when its value is above the threshold.
NULL_RATE_THRESHOLDS = {
    "npi": 0.0,
    "brand_name": 0.0,
    "generic_name": 0.0,
    "total_claims": 0.0,
    "total_drug_cost": 0.0,
    "specialty": 0.01,
}


def standardize_columns(df: DataFrame) -> DataFrame:
    """Keep the columns we use and rename them to snake_case."""
    return df.select([F.col(raw).alias(clean) for raw, clean in COLUMN_MAP.items()])


def cast_types(df: DataFrame) -> DataFrame:
    """Cast numeric columns. Unparseable values become null (and are dropped later).

    try_cast returns null for bad values instead of failing the job. Databricks
    serverless runs with ANSI mode on, where a plain cast() would raise.
    """
    for column in INTEGER_COLUMNS + DOUBLE_COLUMNS:
        df = df.withColumn(column, F.regexp_replace(F.trim(F.col(column)), ",", ""))
    for column in INTEGER_COLUMNS:
        # Cast through double so values like "12.0" still parse.
        df = df.withColumn(column, F.expr(f"CAST(try_cast({column} AS DOUBLE) AS BIGINT)"))
    for column in DOUBLE_COLUMNS:
        df = df.withColumn(column, F.expr(f"try_cast({column} AS DOUBLE)"))
    return df


def standardize_text(df: DataFrame) -> DataFrame:
    """Trim text, turn empty strings into nulls, and upper-case drug names.

    Drug names are upper-cased and whitespace-collapsed so that
    "Atorvastatin Calcium" and "ATORVASTATIN  CALCIUM" compare equal.
    That matters because a drug is generic when brand_name == generic_name.
    """
    df = df.withColumn("npi", F.trim(F.col("npi")))
    for column in TEXT_COLUMNS + DRUG_NAME_COLUMNS:
        cleaned = F.regexp_replace(F.trim(F.col(column)), r"\s+", " ")
        df = df.withColumn(column, F.when(cleaned == "", None).otherwise(cleaned))
    for column in DRUG_NAME_COLUMNS + ["state"]:
        df = df.withColumn(column, F.upper(F.col(column)))
    return df


def drop_invalid_rows(df: DataFrame) -> DataFrame:
    """Drop rows that cannot be used downstream."""
    return df.filter(
        F.col("npi").rlike(r"^\d{10}$")  # NPIs are exactly 10 digits
        & F.col("brand_name").isNotNull()
        & F.col("generic_name").isNotNull()
        & (F.col("total_claims") > 0)
        & (F.col("total_drug_cost") >= 0)
    )


def run_quality_checks(df: DataFrame, stage: str) -> list[dict]:
    """Compute null rates, negative costs and duplicate keys; log any failures.

    Returns one dict per check so the caller can save results to a Delta table.
    """
    total = df.count()
    results = []

    null_counts = df.agg(
        *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in NULL_RATE_THRESHOLDS]
    ).first()
    for column, threshold in NULL_RATE_THRESHOLDS.items():
        rate = (null_counts[column] or 0) / total if total else 0.0
        results.append(_result(stage, f"null_rate:{column}", rate, threshold))

    negative_costs = df.filter(F.col("total_drug_cost") < 0).count()
    results.append(_result(stage, "negative_cost_rows", negative_costs, 0))

    duplicate_keys = df.groupBy(KEY_COLUMNS).count().filter(F.col("count") > 1).count()
    results.append(_result(stage, "duplicate_keys", duplicate_keys, 0))

    for r in results:
        if r["passed"]:
            logger.info("DQ pass [%s] %s = %s", stage, r["check"], r["value"])
        else:
            logger.warning(
                "DQ FAIL [%s] %s = %s (threshold %s)",
                stage, r["check"], r["value"], r["threshold"],
            )
    return results


def _result(stage: str, check: str, value: float, threshold: float) -> dict:
    return {
        "stage": stage,
        "check": check,
        "value": float(value),
        "threshold": float(threshold),
        "passed": value <= threshold,
    }


def build_silver(bronze: DataFrame, year: int) -> tuple[DataFrame, list[dict]]:
    """Full bronze -> silver transformation. Returns (silver_df, dq_results)."""
    typed = standardize_text(cast_types(standardize_columns(bronze)))
    dq_results = run_quality_checks(typed, stage="before_cleaning")
    silver = drop_invalid_rows(typed).dropDuplicates().withColumn("year", F.lit(year))
    return silver, dq_results
