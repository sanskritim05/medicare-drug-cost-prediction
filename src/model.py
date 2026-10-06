"""Predict whether a prescriber is in next year's top 10% of brand-name cost.

Compares Spark ML logistic regression and gradient-boosted trees against a
simple baseline (rank by this year's brand cost), and logs each run to MLflow.
"""

import math

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Heavy-tailed dollar and count features are log-transformed so logistic
# regression is not dominated by a few huge prescribers. Trees are unaffected.
LOG_FEATURES = ["total_claims", "total_cost", "cost_per_claim", "brand_claims", "brand_cost", "distinct_drugs"]
SHARE_FEATURES = ["brand_claim_share", "brand_cost_share"]
NUMERIC_FEATURES = [f"log_{c}" for c in LOG_FEATURES] + SHARE_FEATURES

MODEL_CONFIGS = {
    "logistic_regression": (LogisticRegression, {"maxIter": 100, "regParam": 0.01}),
    "gradient_boosted_trees": (GBTClassifier, {"maxIter": 50, "maxDepth": 5, "seed": 42}),
}

TOP_FRACTION = 0.10


def prepare_features(df: DataFrame) -> DataFrame:
    """Add log features and fill gaps so every row can be scored."""
    for column in LOG_FEATURES:
        df = df.withColumn(f"log_{column}", F.log1p(F.col(column).cast("double")))
    df = df.fillna(0.0, subset=NUMERIC_FEATURES)
    return df.fillna({"specialty": "Unknown"})


def split_train_test(df: DataFrame, test_percent: int = 20) -> tuple[DataFrame, DataFrame]:
    """Deterministic split on a hash of the NPI.

    Unlike randomSplit, this gives the same split every time the DataFrame is
    recomputed, which matters on serverless compute where we cannot cache.
    """
    bucket = F.pmod(F.hash("npi"), F.lit(100))
    return df.filter(bucket >= test_percent), df.filter(bucket < test_percent)


def build_pipeline(classifier) -> Pipeline:
    indexer = StringIndexer(inputCol="specialty", outputCol="specialty_idx", handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=["specialty_idx"], outputCols=["specialty_vec"])
    assembler = VectorAssembler(inputCols=NUMERIC_FEATURES + ["specialty_vec"], outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features")
    return Pipeline(stages=[indexer, encoder, assembler, scaler, classifier])


def precision_at_top_fraction(
    scored: DataFrame, score_col: str = "score", label_col: str = "label", fraction: float = TOP_FRACTION
) -> float:
    """Share of true positives among the highest-scoring `fraction` of rows.

    This is the business metric: if we review the top 10% flagged prescribers,
    how many of them really end up in next year's top 10%?
    """
    n = scored.count()
    k = max(1, math.ceil(n * fraction))
    top_k = scored.orderBy(F.desc(score_col)).limit(k)
    return float(top_k.agg(F.avg(label_col)).first()[0])


def evaluate(scored: DataFrame, score_col: str = "score") -> dict:
    auroc = BinaryClassificationEvaluator(
        rawPredictionCol=score_col, labelCol="label", metricName="areaUnderROC"
    ).evaluate(scored)
    return {"auroc": auroc, "precision_at_top_10pct": precision_at_top_fraction(scored, score_col)}


def run_experiment(training_data: DataFrame, run_params: dict) -> list[dict]:
    """Train and evaluate the baseline and both models, one MLflow run each.

    `training_data` is year-N gold features with a year-N+1 `label` column.
    `run_params` (years, states, ...) are logged on every run for context.
    """
    import mlflow  # imported here so the rest of the module works without MLflow

    data = prepare_features(training_data)
    train, test = split_train_test(data)
    context = {
        **run_params,
        "n_train": train.count(),
        "n_test": test.count(),
        "test_positive_rate": test.agg(F.avg("label")).first()[0],
    }
    results = []

    # Baseline: assume last year's biggest brand spenders stay on top.
    with mlflow.start_run(run_name="baseline_prior_year_brand_cost"):
        mlflow.log_params({**context, "model": "baseline"})
        metrics = evaluate(test.withColumn("score", F.col("brand_cost").cast("double")))
        mlflow.log_metrics(metrics)
        results.append({"model": "baseline", **metrics})

    for name, (estimator_class, params) in MODEL_CONFIGS.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({**context, **params, "model": name})
            fitted = build_pipeline(estimator_class(**params)).fit(train)
            scored = fitted.transform(test).withColumn(
                "score", vector_to_array("probability")[1]
            )
            metrics = evaluate(scored)
            mlflow.log_metrics(metrics)
            results.append({"model": name, **metrics})

    return results
