# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver
# MAGIC Rename to snake_case, cast numbers, standardize text, drop invalid rows and exact duplicates.
# MAGIC Data quality checks run twice: on the typed data **before** cleaning (to measure the raw
# MAGIC problems) and on the silver output **after** cleaning (to confirm they are gone).
# MAGIC Failures are logged as warnings and every result is appended to `silver_dq_results`.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

from pyspark.sql import functions as F

from src.pipeline import run_silver

for summary in run_silver(spark, YEARS, bronze_table, SILVER_TABLE, DQ_TABLE):
    print(
        f"{summary['year']}: "
        f"{summary['bronze_rows']:,} bronze rows -> "
        f"{summary['silver_rows']:,} silver rows"
    )

# COMMAND ----------

display(
    spark.table(DQ_TABLE)
    .filter(F.col("year").isin(YEARS))
    .orderBy(F.desc("checked_at"), "year", "stage", "check")
)
