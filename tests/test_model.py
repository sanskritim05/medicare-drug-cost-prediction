import pytest

from src.model import precision_at_top_fraction, split_train_test


def test_precision_at_top_fraction(spark):
    # 20 rows, so the top 10% is the 2 highest scores: one positive, one negative.
    rows = [(float(s), 0) for s in range(18)] + [(100.0, 1), (99.0, 0)]
    scored = spark.createDataFrame(rows, ["score", "label"])
    assert precision_at_top_fraction(scored) == pytest.approx(0.5)


def test_split_is_disjoint_and_repeatable(spark):
    df = spark.createDataFrame([(str(i),) for i in range(200)], ["npi"])
    train, test = split_train_test(df)
    train_ids = {r["npi"] for r in train.collect()}
    test_ids = {r["npi"] for r in test.collect()}

    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == 200
    assert test_ids == {r["npi"] for r in split_train_test(df)[1].collect()}
