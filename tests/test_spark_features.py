"""Runs features.sql in a local Spark session on a tiny hand-made dataset."""

import shutil

import pytest

pytest.importorskip("pyspark")
if shutil.which("java") is None:
    import os
    if not os.environ.get("JAVA_HOME"):
        pytest.skip("Java is not available", allow_module_level=True)

from pvm.spark_features import build_features, spark_session  # noqa: E402

PLAYERS = (
    '"UserID","CountryID","Gender","BirthYear","DateReg","TimeReg","Date1Dep","Date1Bet",'
    '"Date1Spo","StakeF","StakeL","StakeA","WinF","WinL","WinA","BetsF","BetsL","BetsA",'
    '"DaysF","DaysL","DaysA"\n'
    '1,276,1,1980,2005-02-01,"00:01",2005-02-01,2005-02-02,2005-02-02,0,0,0,0,0,0,0,0,0,0,0,0\n'
    '2,300,0,NA,2005-02-01,"00:02",2005-02-03,2005-02-03,2005-02-03,0,0,0,0,0,0,0,0,0,0,0,0\n'
)
# Player 1: a settlement-only day (win, no stake) before the first stake, three active
# days in the first 90 days, one live day, and two days in the prediction window.
FIXED = (
    '"UserID","DateBet","StakeF","WinF","BetsF"\n'
    "1,2005-02-01,0,5,0\n"
    "1,2005-02-02,10,0,2\n"
    "1,2005-02-05,20,30,1\n"
    "1,2005-05-01,30,0,3\n"
    "1,2005-06-01,100,50,4\n"
    "1,2006-01-10,40,0,1\n"
    "2,2005-02-03,5,0,1\n"
)
LIVE = (
    '"UserID","DateBet","StakeL","WinL","BetsL"\n'
    "1,2005-02-05,8,0,2\n"
)


@pytest.fixture(scope="module")
def feats(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    (d / "PopTrendsBData3Aggs.csv").write_text(PLAYERS)
    (d / "PopTrendsBData1Fixed.csv").write_text(FIXED)
    (d / "PopTrendsBData2Live.csv").write_text(LIVE)
    spark = spark_session("test")
    out = build_features(spark, d).toPandas().set_index("UserID")
    spark.stop()
    return out


def test_day_zero_is_first_stake_not_settlement(feats):
    assert str(feats.loc[1, "t0"]) == "2005-02-02"


def test_observation_window_features(feats):
    p = feats.loc[1]
    assert p["stake_90"] == pytest.approx(10 + 20 + 8 + 30)   # Feb 2, Feb 5 (+live), May 1
    assert p["stake_live_90"] == pytest.approx(8)
    assert p["active_days_90"] == 3
    assert p["last_active_day"] == 88                         # 2005-05-01 is day 88
    assert p["mean_gap"] == pytest.approx((3 + 85) / 2)
    assert p["max_7d_stake"] == pytest.approx(38)             # Feb 2 + Feb 5 inside 7 days
    assert p["live_days_90"] == 1
    assert p["reg_to_first_bet"] == 1


def test_targets_use_days_90_to_454_only(feats):
    p = feats.loc[1]
    assert p["y_turnover"] == pytest.approx(100 + 40)
    assert p["y_ggr"] == pytest.approx(140 - 50)
    assert p["y_active_days"] == 2
    assert p["y_retained_90"] == 1                            # June 1 is day 119


def test_missing_birth_year_and_no_future_activity(feats):
    p = feats.loc[2]
    assert p["BirthYear"] != p["BirthYear"]                   # NaN from 'NA'
    assert p["y_turnover"] == 0 and p["y_retained_90"] == 0
