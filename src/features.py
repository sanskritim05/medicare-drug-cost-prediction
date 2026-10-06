"""Gold layer: one row per prescriber per year, plus the next-year label."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Matches CMS generic names of vaccines, e.g. "HPV VACCINE 9-VALENT/PF",
# "DIPHTH,PERTUSS(ACELL),TET VAC", "TETANUS-DIPHTHERIA TOXOIDS/PF",
# "RSVPREF3 ANTIGEN/AS01E/PF" and "VARICELLA-ZOSTER GE/AS01B/PF" (Shingrix).
VACCINE_PATTERN = r"\bVACC?(INE)?\b|\bTOX(OIDS?)?\b|ANTIGEN|AS01"


def add_is_generic(silver: DataFrame) -> DataFrame:
    """A drug row is generic when its brand name equals its generic name.

    Names were upper-cased and whitespace-normalized in silver, so this is an
    exact comparison.
    """
    return silver.withColumn("is_generic", F.col("brand_name") == F.col("generic_name"))


def add_is_vaccine(silver: DataFrame) -> DataFrame:
    """Flag vaccines by their generic name.

    Vaccines have no generic version, so the brand/generic rule always counts
    them as brand. They are left out of brand totals because a vaccine is never
    a generic-substitution opportunity, and pharmacy standing-order prescribers
    would otherwise top the brand-cost ranking.
    """
    return silver.withColumn("is_vaccine", F.col("generic_name").rlike(VACCINE_PATTERN))


def _safe_divide(numerator: F.Column, denominator: F.Column) -> F.Column:
    return F.when(denominator > 0, numerator / denominator)


def build_prescriber_features(silver: DataFrame) -> DataFrame:
    """Aggregate prescriber x drug rows into one row per prescriber per year."""
    df = add_is_vaccine(add_is_generic(silver))
    # Vaccines still count toward total claims and total cost.
    brand = ~F.col("is_generic") & ~F.col("is_vaccine")

    totals = df.groupBy("npi", "year").agg(
        # A prescriber has one specialty in practice; mode() guards against
        # the rare prescriber listed under more than one.
        F.mode("specialty").alias("specialty"),
        F.mode("state").alias("state"),
        F.sum("total_claims").alias("total_claims"),
        F.sum("total_drug_cost").alias("total_cost"),
        F.sum(F.when(brand, F.col("total_claims")).otherwise(0)).alias("brand_claims"),
        F.sum(F.when(brand, F.col("total_drug_cost")).otherwise(0.0)).alias("brand_cost"),
        # Count active ingredients, so a molecule sold as brand and generic counts once.
        F.countDistinct("generic_name").alias("distinct_drugs"),
    )

    return totals.select(
        "npi",
        "year",
        "specialty",
        "state",
        "total_claims",
        "total_cost",
        _safe_divide(F.col("total_cost"), F.col("total_claims")).alias("cost_per_claim"),
        "brand_claims",
        "brand_cost",
        _safe_divide(F.col("brand_claims"), F.col("total_claims")).alias("brand_claim_share"),
        _safe_divide(F.col("brand_cost"), F.col("total_cost")).alias("brand_cost_share"),
        "distinct_drugs",
    )


def add_next_year_label(
    features_year_n: DataFrame, features_year_n1: DataFrame, top_fraction: float = 0.10
) -> DataFrame:
    """Label each year-N prescriber by whether they are in the top 10% of
    brand-name cost (excluding vaccines) in year N+1.

    The cutoff is the 90th percentile of brand_cost among year N+1 prescribers,
    and the label is brand_cost strictly above it. Prescribers missing from
    year N+1 had no qualifying Part D claims that year, so they get label 0.
    """
    cutoff = features_year_n1.agg(
        F.percentile_approx("brand_cost", 1 - top_fraction, 100000).alias("cutoff")
    ).first()["cutoff"]

    next_year = features_year_n1.select(
        "npi", (F.col("brand_cost") > F.lit(cutoff)).cast("int").alias("label")
    )
    return features_year_n.join(next_year, on="npi", how="left").fillna({"label": 0})
