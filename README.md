# Predicting high brand-name drug spenders in Medicare Part D

A PySpark pipeline on Databricks Free Edition. It turns raw CMS Medicare Part D prescriber data into a
bronze/silver/gold Delta lakehouse, then predicts which prescribers will be in the
**top 10% of brand-name drug cost next year**.

## The problem

Brand-name drugs are a small share of Medicare Part D claims but a large share of its cost.
A plan or a program-integrity team with limited review capacity wants to know, ahead of time,
which prescribers are likely to be among the highest brand-name spenders. Those prescribers are
candidates for formulary outreach or generic-substitution education.

**Task:** using a prescriber's year *N* prescribing profile, predict whether they will be in the
top 10% of brand-name drug cost in year *N+1*.

**Data:** CMS [Medicare Part D Prescribers - by Provider and Drug](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug).
It has one row per prescriber (NPI) × drug (brand name + generic name) per year, with claims, day supply,
total drug cost and beneficiary counts.

## Architecture

```
data.cms.gov CSV (one file per year)
        │  upload to Unity Catalog volume  /Volumes/workspace/partd/raw
        ▼
BRONZE  bronze_prescriber_drug_<year>   raw copy, all columns as strings
        │  rename · cast · standardize · DQ checks · drop invalid · dedupe
        ▼
SILVER  silver_prescriber_drug          typed prescriber × drug rows, `year` column
        silver_dq_results               one row per data quality check per run
        │  aggregate per prescriber
        ▼
GOLD    gold_prescriber_year            one row per prescriber per year
        │  year N features + year N+1 label
        ▼
MODEL   baseline vs logistic regression vs gradient-boosted trees, tracked in MLflow
```

| Layer | What it does | Why |
|---|---|---|
| Bronze | Reads the CSV with `inferSchema=false` and writes it unchanged to Delta. | You can rebuild everything downstream without re-downloading, and you always have the source as delivered. |
| Silver | Renames columns to snake_case. Casts numbers with `try_cast`. Trims text and upper-cases drug names. Drops rows with a bad NPI, missing drug names, no claims or negative cost. Removes exact duplicates. | Typing and validation happen once, in one tested place. |
| Gold | One row per prescriber per year: total cost, cost per claim, brand share of claims and of cost, distinct drug count, specialty. | This is the grain the model and any dashboard need. |

**Data quality checks** (`src/cleaning.py::run_quality_checks`) run twice: on the typed data before
cleaning, which measures the raw problems, and on the silver table after cleaning, which confirms they are gone:

- null rate of key columns (NPI, drug names, claims, cost: 0% allowed; specialty: 1%)
- rows with negative drug cost (0 allowed)
- duplicate `(npi, brand_name, generic_name)` keys (0 allowed)

A failed check logs a `WARNING` and is saved to `silver_dq_results` with `passed = false`.
Checks log rather than stop the job. Because the bad rows are dropped, one bad row should not block a whole year.
You can query the table to see quality over time.

**Generic vs brand rule:** a drug row is generic when `brand_name == generic_name`. Silver upper-cases
names and collapses whitespace first, so "Atorvastatin Calcium" and "ATORVASTATIN  CALCIUM" match.

**Rerunnable writes:** silver and gold are single tables with a `year` column. Each year is written with
Delta `replaceWhere = "year = N"`, so rerunning one year replaces that year's rows without
duplicating or touching other years.

## Model

- **Features (year N):** log of total claims, total cost, cost per claim, brand claims, brand cost and distinct
  drugs; brand claim share; brand cost share; one-hot specialty.
- **Label (year N+1):** 1 if the prescriber's brand cost is above the 90th percentile among year N+1
  prescribers. Prescribers who do not appear in year N+1 get 0.
- **Split:** 80/20 by a hash of the NPI. It is deterministic, so the split stays the same when Spark recomputes the
  DataFrame. That matters because serverless compute does not allow `.cache()`.
- **Models:**
  - *Baseline:* rank by year N brand cost. This is the "last year's big spenders stay big" rule.
  - Spark ML **logistic regression** (`regParam=0.01`)
  - Spark ML **gradient-boosted trees** (50 trees, depth 5)
- **Metrics:**
  - **AUROC:** how well the model ranks prescribers overall.
  - **Precision at top 10%:** of the 10% of prescribers the model ranks highest, the share that really are in next year's top 10%.
    This is the number a review team with fixed capacity cares about.
- **Tracking:** each of the three runs is logged to the notebook's MLflow experiment with its params
  (years, states, row counts, hyperparameters) and metrics.

## Results

> Fill these in after running `notebooks/04_model.py`. The numbers come from the MLflow runs.

States: CT, RI, DE · Features: 2022 · Label: 2023 · Test prescribers: _TBD_ · Positive rate: _TBD_

| Model | AUROC | Precision at top 10% |
|---|---|---|
| Baseline (prior-year brand cost) | _TBD_ | _TBD_ |
| Logistic regression | _TBD_ | _TBD_ |
| Gradient-boosted trees | _TBD_ | _TBD_ |

Brand-name spending is very persistent from one year to the next, so expect the baseline to be strong. The question
is how much the models add on top of it.

## Limitations

- **States subset.** To fit Free Edition compute limits, bronze loads only the states in
  `notebooks/config.py` (CT, RI, DE by default). The top-10% cutoff is computed within those states,
  not nationally. A prescriber who moves to another state looks like they dropped out (label 0).
  Set `STATES = None` to process the whole country on larger compute.
- **One pair of years, no out-of-time test.** Train and test both come from the same 2022→2023 pair.
  A stronger check would train on 2021→2022 and test on 2022→2023.
- **CMS suppression.** CMS omits prescriber × drug rows with fewer than 11 claims, so small prescribers'
  totals are understated and some prescribers are missing entirely.
- **Brand/generic rule is a heuristic.** `brand_name == generic_name` misclassifies some branded
  generics and biosimilars.
- **Gross cost.** `Tot_Drug_Cst` is cost before manufacturer rebates, so it overstates what Medicare
  really pays for many brand drugs.
- **Specialty** is the CMS-derived prescriber type, not verified credentials.
- **No hyperparameter tuning, and no model artifact logged.** MLflow tracks params and metrics only.
  Logging Spark ML models on serverless needs extra setup (a UC volume temp dir), which this project skips.
- **Not causal.** A high score means "looks like past high spenders". It does not mean "prescribing inappropriately".

## How to run

### On Databricks Free Edition

1. **Add the repo.** In Databricks, go to *Workspace → Create → Git folder* and paste this repo's URL.
2. **Set up storage.** Open `notebooks/00_setup` and run it. It creates schema `workspace.partd` and volume `raw`.
3. **Get the data.** Download the CSV for 2022 and 2023 from data.cms.gov. Upload each to the
   `raw` volume as `partd_prescriber_drug_<year>.csv`, or paste the download links into the last cell of
   `00_setup`.
4. **Run the pipeline.** Run `01_bronze`, then `02_silver`, `03_gold` and `04_model`, in order.
   - Years and states are set in `notebooks/config.py`.
   - Every notebook loads the config with `%run ./config`.
5. **Check the runs.** Open the notebook's *Experiment* panel to compare the MLflow runs.

Notes for serverless compute:
- Code uses `try_cast` because ANSI mode is on and plain `cast` would fail on bad values.
- It never calls `.cache()` or the RDD API, which serverless compute does not support.
- If `pyspark.ml` fails to import, pick the newest serverless environment version in the notebook's
  *Environment* side panel.

### Tests locally

You need Python 3.10–3.12 and Java 17 or 21.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -v
```

The tests use a local Spark session and small hand-built fixtures, so they need no Databricks access and no
data download. GitHub Actions runs the same tests on every push (`.github/workflows/tests.yml`).

## Project structure

```
src/
  ingestion.py     bronze: read raw CSV, optional state filter, Delta writes
  cleaning.py      silver: rename, cast, standardize, validate, data quality checks
  features.py      gold: per-prescriber features and the next-year label
  model.py         baseline, logistic regression and GBT, metrics, MLflow runs
  pipeline.py      notebook orchestration helpers for bronze/silver/gold/model steps
notebooks/         Databricks notebooks (source format) that call src/
  config.py        catalog, schema, years, states
  00_setup.py  01_bronze.py  02_silver.py  03_gold.py  04_model.py
tests/             pytest + local PySpark unit tests
.github/workflows/tests.yml
```

The logic lives in `src/` as plain functions. The notebooks only load config, call one pipeline helper,
and display results. That keeps transformations unit-testable locally and the Databricks notebooks easy to scan.
