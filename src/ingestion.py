"""Bronze layer: land the raw CMS CSV in Delta exactly as delivered.

Every column is read as a string (no schema inference), so the bronze table is
a faithful copy of the source file. Typing and cleaning happen in silver.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

RAW_STATE_COLUMN = "Prscrbr_State_Abrvtn"


def read_raw_csv(spark: SparkSession, path: str) -> DataFrame:
    """Read the CMS CSV with every column as a string."""
    return (
        spark.read.option("header", True)
        .option("inferSchema", False)  # keep raw values; types are decided in silver
        .option("quote", '"')
        .option("escape", '"')  # CMS escapes quotes inside names by doubling them
        .csv(path)
    )


def filter_states(df: DataFrame, states: list[str] | None) -> DataFrame:
    """Keep only rows for the given state codes. `None` keeps everything.

    This exists only to fit Databricks Free Edition compute limits. It removes
    rows but never changes a value, so bronze stays a raw copy of those rows.
    """
    if not states:
        return df
    return df.filter(F.col(RAW_STATE_COLUMN).isin(states))


def write_delta(df: DataFrame, table_name: str, replace_where: str | None = None) -> None:
    """Overwrite a Delta table, or only the rows matching `replace_where`.

    `replace_where` (e.g. "year = 2022") makes reruns idempotent: rerunning one
    year replaces that year's rows and leaves the other years alone.
    """
    writer = df.write.format("delta").mode("overwrite")
    if replace_where and df.sparkSession.catalog.tableExists(table_name):
        writer = writer.option("replaceWhere", replace_where)
    else:
        writer = writer.option("overwriteSchema", True)
    writer.saveAsTable(table_name)


def ingest_year(
    spark: SparkSession, csv_path: str, table_name: str, states: list[str] | None = None
) -> int:
    """Load one year's raw CSV into a bronze Delta table. Returns the row count."""
    raw = filter_states(read_raw_csv(spark, csv_path), states)
    write_delta(raw, table_name)
    return spark.table(table_name).count()
