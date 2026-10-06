# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 · Gold
# MAGIC One row per prescriber per year: total cost, cost per claim, brand-name share of claims
# MAGIC and cost, distinct drug count and specialty. A drug is generic when its brand name equals
# MAGIC its generic name.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import src.features as f
print(f.__file__)
print("has vaccine rule:", hasattr(f, "VACCINE_PATTERN"))


# COMMAND ----------

from pyspark.sql import functions as F

from src.pipeline import run_gold

run_gold(spark, YEARS, SILVER_TABLE, GOLD_TABLE)

display(spark.table(GOLD_TABLE).groupBy("year").count().orderBy("year"))

# COMMAND ----------

display(
    spark.table(GOLD_TABLE)
    .filter(F.col("year") == FEATURE_YEAR)
    .orderBy(F.desc("brand_cost"))
    .limit(20)
)