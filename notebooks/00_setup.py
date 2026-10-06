# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the schema and a Unity Catalog volume to hold the raw CSV files.
# MAGIC
# MAGIC **Getting the data.** On data.cms.gov, open *Medicare Part D Prescribers - by Provider and Drug*,
# MAGIC pick a year, and download the CSV. Do this for both `FEATURE_YEAR` and `LABEL_YEAR` (see `config`).
# MAGIC Each file is several GB, so either:
# MAGIC 1. Upload it in the UI: **Catalog → workspace → partd → raw → Upload to this volume**, or
# MAGIC 2. Paste the CSV download link into the cell at the bottom to download it straight into the volume.
# MAGIC
# MAGIC Name each file `partd_prescriber_drug_<year>.csv`.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print("Raw files go in:", RAW_DIR)

# COMMAND ----------

# Optional: download a CSV straight into the volume.
# Paste the link from the "Download" button on data.cms.gov for each year.
import shutil
import urllib.request

DOWNLOAD_URLS = {
    # 2022: "https://data.cms.gov/.../....csv",
    # 2023: "https://data.cms.gov/.../....csv",
}

for year, url in DOWNLOAD_URLS.items():
    target = raw_csv_path(year)
    print(f"Downloading {year} -> {target}")
    with urllib.request.urlopen(url) as response, open(target, "wb") as out:
        shutil.copyfileobj(response, out, length=16 * 1024 * 1024)

# COMMAND ----------

display(dbutils.fs.ls(RAW_DIR))

# COMMAND ----------

print(open("/Volumes/workspace/partd/raw/partd_prescriber_drug_2023.csv").readline())
