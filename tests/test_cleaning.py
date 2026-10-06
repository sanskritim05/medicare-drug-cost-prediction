import logging

from pyspark.sql import functions as F

from src.cleaning import (
    COLUMN_MAP,
    build_silver,
    cast_types,
    drop_invalid_rows,
    run_quality_checks,
    standardize_columns,
    standardize_text,
)
from tests.conftest import raw_row


def _typed(make_raw_df, rows):
    return standardize_text(cast_types(standardize_columns(make_raw_df(rows))))


def _checks_by_name(results):
    return {r["check"]: r for r in results}


def test_standardize_columns_renames_and_drops_extras(make_raw_df):
    df = standardize_columns(make_raw_df([raw_row()]))
    assert df.columns == list(COLUMN_MAP.values())
    assert "GE65_Tot_Clms" not in df.columns


def test_cast_types_parses_numbers_and_nulls_bad_values(make_raw_df):
    rows = [
        raw_row(Tot_Clms="1,234", Tot_Drug_Cst="12.50"),
        raw_row(Tot_Clms="abc", Tot_Drug_Cst=""),
        raw_row(Tot_Clms=" 15.0 ", Tot_Drug_Cst=None),
    ]
    out = cast_types(standardize_columns(make_raw_df(rows))).collect()

    assert (out[0]["total_claims"], out[0]["total_drug_cost"]) == (1234, 12.5)
    assert (out[1]["total_claims"], out[1]["total_drug_cost"]) == (None, None)
    assert (out[2]["total_claims"], out[2]["total_drug_cost"]) == (15, None)


def test_standardize_text_normalizes_names(make_raw_df):
    rows = [
        raw_row(
            Brnd_Name="  atorvastatin   calcium ",
            Gnrc_Name="Atorvastatin Calcium",
            Prscrbr_State_Abrvtn="ri",
            Prscrbr_City="   ",
        )
    ]
    out = _typed(make_raw_df, rows).first()

    assert out["brand_name"] == "ATORVASTATIN CALCIUM"
    assert out["generic_name"] == "ATORVASTATIN CALCIUM"
    assert out["state"] == "RI"
    assert out["city"] is None  # blank strings become null


def test_drop_invalid_rows_keeps_only_valid(make_raw_df):
    rows = [
        raw_row(Prscrbr_NPI="1111111111"),                  # valid
        raw_row(Prscrbr_NPI="123"),                         # NPI too short
        raw_row(Prscrbr_NPI="2222222222", Brnd_Name=""),    # missing brand name
        raw_row(Prscrbr_NPI="3333333333", Tot_Clms="0"),    # no claims
        raw_row(Prscrbr_NPI="4444444444", Tot_Drug_Cst="-5"),  # negative cost
        raw_row(Prscrbr_NPI="5555555555", Tot_Drug_Cst="0"),   # zero cost is allowed
    ]
    out = drop_invalid_rows(_typed(make_raw_df, rows))
    assert sorted(r["npi"] for r in out.collect()) == ["1111111111", "5555555555"]


def test_quality_checks_pass_on_clean_data(make_raw_df):
    rows = [raw_row(Prscrbr_NPI="1111111111"), raw_row(Prscrbr_NPI="2222222222")]
    results = run_quality_checks(_typed(make_raw_df, rows), stage="test")
    assert all(r["passed"] for r in results)


def test_quality_checks_flag_and_log_failures(make_raw_df, caplog):
    rows = [
        raw_row(Prscrbr_NPI="1111111111"),
        raw_row(Prscrbr_NPI="1111111111"),                 # duplicate key
        raw_row(Prscrbr_NPI="2222222222", Tot_Drug_Cst="-10"),  # negative cost
        raw_row(Prscrbr_NPI="3333333333", Tot_Clms="n/a"),      # null claims
    ]
    with caplog.at_level(logging.WARNING):
        results = _checks_by_name(run_quality_checks(_typed(make_raw_df, rows), stage="test"))

    assert results["duplicate_keys"]["value"] == 1
    assert results["negative_cost_rows"]["value"] == 1
    assert results["null_rate:total_claims"]["value"] == 0.25
    assert not results["duplicate_keys"]["passed"]
    assert not results["negative_cost_rows"]["passed"]
    assert not results["null_rate:total_claims"]["passed"]
    assert results["null_rate:npi"]["passed"]
    assert "DQ FAIL [test] duplicate_keys" in caplog.text


def test_build_silver_end_to_end(make_raw_df):
    rows = [
        raw_row(Prscrbr_NPI="1111111111"),
        raw_row(Prscrbr_NPI="1111111111"),       # exact duplicate, removed
        raw_row(Prscrbr_NPI="bad"),              # invalid, removed
    ]
    silver, dq_results = build_silver(make_raw_df(rows), year=2022)

    assert silver.count() == 1
    assert silver.select(F.col("year")).first()["year"] == 2022
    assert any(r["check"] == "duplicate_keys" and not r["passed"] for r in dq_results)
