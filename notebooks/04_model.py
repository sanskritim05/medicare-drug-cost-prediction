# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04 · Model
# MAGIC Using `FEATURE_YEAR` features, predict whether a prescriber is in the top 10% of brand-name
# MAGIC cost in `LABEL_YEAR`. Compares a naive baseline, logistic regression and gradient-boosted trees
# MAGIC on AUROC and precision at the top 10%. Each run is logged to this notebook's MLflow experiment.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

from src.pipeline import run_model

results = run_model(spark, GOLD_TABLE, FEATURE_YEAR, LABEL_YEAR, STATES)

# COMMAND ----------

display(spark.createDataFrame(results))