# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 · Bronze
# MAGIC Load each year's raw CSV into its own Delta table with every column kept as a string.
# MAGIC No renaming, no typing, no cleaning. Bronze is a replayable copy of the source.
# MAGIC The only deviation is the optional state filter in `config` (Free Edition compute limits).

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

from src.pipeline import run_bronze

for summary in run_bronze(spark, YEARS, raw_csv_path, bronze_table, STATES):
    print(f"{summary['table']}: {summary['rows']:,} rows")

# COMMAND ----------

display(spark.table(bronze_table(FEATURE_YEAR)).limit(10))