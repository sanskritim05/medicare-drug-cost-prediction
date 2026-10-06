import pytest

from src.features import (
    add_is_generic,
    add_is_vaccine,
    add_next_year_label,
    build_prescriber_features,
)

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


# Generic names as they appear in the 2023 CMS file.
VACCINE_GENERIC_NAMES = [
    "RSV VACC, PREF A AND PREF B/PF",
    "DIPH,PERTUSS(ACELL),TET VAC/PF",
    "RSVPREF3 ANTIGEN/AS01E/PF",
    "HEPATITIS B VIRUS VACCINE/PF",
    "MENING VAC A,C,Y,W135,C-TET/PF",
    "VARICELLA-ZOSTER GE/AS01B/PF",
    "TETANUS, DIPHTHERIA TOX,ADULT",
    "TETANUS-DIPHTHERIA TOXOIDS/PF",
    "MEASLES,MUMPS,RUBELLA VACC/PF",
]
# Non-vaccines, including look-alikes that contain "VAC" or "TOX" inside a word.
NON_VACCINE_GENERIC_NAMES = [
    "ATORVASTATIN CALCIUM",
    "SEMAGLUTIDE",
    "INSULIN GLARGINE,HUM.REC.ANLOG",
    "ONABOTULINUMTOXINA",
    "AVACOPAN",
    "VALACYCLOVIR HCL",
]


def test_add_is_vaccine(spark):
    names = VACCINE_GENERIC_NAMES + NON_VACCINE_GENERIC_NAMES
    df = spark.createDataFrame([(n,) for n in names], ["generic_name"])
    flags = {r["generic_name"]: r["is_vaccine"] for r in add_is_vaccine(df).collect()}

    assert [n for n in VACCINE_GENERIC_NAMES if not flags[n]] == []
    assert [n for n in NON_VACCINE_GENERIC_NAMES if flags[n]] == []


def test_vaccines_count_in_totals_but_not_brand(spark):
    rows = [
        ("V", 2023, "Internal Medicine", "CT", "SHINGRIX", "VARICELLA-ZOSTER GE/AS01B/PF", 100, 20000.0),
        ("V", 2023, "Internal Medicine", "CT", "ELIQUIS", "APIXABAN", 10, 5000.0),
        ("V", 2023, "Internal Medicine", "CT", "LISINOPRIL", "LISINOPRIL", 10, 100.0),
    ]
    v = build_prescriber_features(spark.createDataFrame(rows, SILVER_COLUMNS)).first()

    assert v["total_claims"] == 120
    assert v["total_cost"] == 25100.0
    assert v["brand_claims"] == 10  # Eliquis only
    assert v["brand_cost"] == 5000.0
    assert v["brand_claim_share"] == pytest.approx(10 / 120)


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
