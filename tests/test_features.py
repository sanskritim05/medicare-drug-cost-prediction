import pytest

from src.features import add_is_generic, add_next_year_label, build_prescriber_features

SILVER_COLUMNS = [
    "npi", "year", "specialty", "state", "brand_name", "generic_name",
    "total_claims", "total_drug_cost",
]


@pytest.fixture
def silver(spark):
    """Two prescribers. Prescriber A has one brand and two generic drugs."""
    rows = [
        # npi, year, specialty, state, brand, generic, claims, cost
        ("A", 2022, "Cardiology", "RI", "LIPITOR", "ATORVASTATIN CALCIUM", 10, 600.0),
        ("A", 2022, "Cardiology", "RI", "ATORVASTATIN CALCIUM", "ATORVASTATIN CALCIUM", 20, 100.0),
        ("A", 2022, "Cardiology", "RI", "METFORMIN HCL", "METFORMIN HCL", 10, 300.0),
        ("B", 2022, "Family Practice", "CT", "ELIQUIS", "APIXABAN", 5, 2500.0),
    ]
    return spark.createDataFrame(rows, SILVER_COLUMNS)


def test_add_is_generic(silver):
    flags = {(r["npi"], r["brand_name"]): r["is_generic"] for r in add_is_generic(silver).collect()}
    assert flags[("A", "LIPITOR")] is False
    assert flags[("A", "ATORVASTATIN CALCIUM")] is True
    assert flags[("B", "ELIQUIS")] is False


def test_build_prescriber_features_one_row_per_prescriber(silver):
    gold = {r["npi"]: r for r in build_prescriber_features(silver).collect()}
    assert set(gold) == {"A", "B"}

    a = gold["A"]
    assert a["specialty"] == "Cardiology"
    assert a["total_claims"] == 40
    assert a["total_cost"] == 1000.0
    assert a["cost_per_claim"] == 25.0
    assert a["brand_claims"] == 10
    assert a["brand_cost"] == 600.0
    assert a["brand_claim_share"] == pytest.approx(0.25)
    assert a["brand_cost_share"] == pytest.approx(0.6)
    assert a["distinct_drugs"] == 2  # atorvastatin (brand + generic) and metformin

    b = gold["B"]
    assert b["brand_claim_share"] == 1.0
    assert b["brand_cost_share"] == 1.0


def test_brand_cost_share_is_null_when_total_cost_is_zero(spark):
    rows = [("C", 2022, "Dentist", "DE", "AMOXICILLIN", "AMOXICILLIN", 12, 0.0)]
    c = build_prescriber_features(spark.createDataFrame(rows, SILVER_COLUMNS)).first()
    assert c["total_cost"] == 0.0
    assert c["brand_cost_share"] is None


def test_add_next_year_label_flags_top_10_percent(spark):
    # Year N+1: ten prescribers with brand cost 1..10; only P10 is in the top 10%.
    next_year = spark.createDataFrame(
        [(f"P{i}", float(i)) for i in range(1, 11)], ["npi", "brand_cost"]
    )
    # Year N: P10 (top next year), P1 (bottom), and P99 (absent next year).
    this_year = spark.createDataFrame(
        [("P10", 5.0), ("P1", 50.0), ("P99", 100.0)], ["npi", "brand_cost"]
    )

    labels = {r["npi"]: r["label"] for r in add_next_year_label(this_year, next_year).collect()}
    assert labels == {"P10": 1, "P1": 0, "P99": 0}
