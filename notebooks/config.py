# Databricks notebook source
# MAGIC %md
# MAGIC # Shared configuration
# MAGIC Every pipeline notebook runs this first with `%run ./config`.
# MAGIC Change the years or states here and nowhere else.

# COMMAND ----------

import logging
import os
import sys

# Make the repo's `src` package importable from notebooks in a Databricks Git folder.
REPO_ROOT = os.path.abspath("..")
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)

# Free Edition ships with the `workspace` catalog.
CATALOG = "workspace"
SCHEMA = "partd"
VOLUME = "raw"

# Features come from FEATURE_YEAR; the label comes from FEATURE_YEAR + 1.
FEATURE_YEAR = 2023
LABEL_YEAR = FEATURE_YEAR + 1
YEARS = [FEATURE_YEAR, LABEL_YEAR]

# Filter to a few states to stay inside Free Edition compute limits.
# Set to None to process the whole country.
STATES = ["CT", "RI", "DE"]

RAW_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_prescriber_drug"
DQ_TABLE = f"{CATALOG}.{SCHEMA}.silver_dq_results"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.gold_prescriber_year"


def raw_csv_path(year: int) -> str:
    return f"{RAW_DIR}/partd_prescriber_drug_{year}.csv"


def bronze_table(year: int) -> str:
    return f"{CATALOG}.{SCHEMA}.bronze_prescriber_drug_{year}"