"""Build the player feature table with PySpark and Spark SQL.

Reads the three Transparency Project CSVs (daily fixed-odds bets, daily live bets,
one row per player), runs sql/features.sql and writes one Parquet file with a row
per player: features from days 0-89 after the first stake, targets from days 90-454.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SQL_PATH = Path(__file__).parent / "sql" / "features.sql"
OBS_DAYS = 90
PRED_DAYS = 365


def spark_session(app: str = "player-value-model") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_views(spark: SparkSession, data_dir: Path) -> None:
    def read(name: str):
        return spark.read.csv(str(data_dir / name), header=True, inferSchema=True)

    read("PopTrendsBData1Fixed.csv").withColumn("DateBet", F.to_date("DateBet")) \
        .createOrReplaceTempView("fixed")
    read("PopTrendsBData2Live.csv").withColumn("DateBet", F.to_date("DateBet")) \
        .createOrReplaceTempView("live")
    players = read("PopTrendsBData3Aggs.csv")
    for c in ("DateReg", "Date1Dep", "Date1Bet", "Date1Spo"):
        players = players.withColumn(c, F.to_date(c))
    players.createOrReplaceTempView("players")


def feature_sql(obs_days: int = OBS_DAYS, pred_days: int = PRED_DAYS) -> str:
    sql = SQL_PATH.read_text(encoding="utf-8")
    return sql.replace("${OBS_DAYS}", str(obs_days)).replace("${PRED_DAYS}", str(pred_days))


def build_features(spark: SparkSession, data_dir: Path, last_date: str = "2006-08-31"):
    """Feature table for players whose full 455-day horizon fits inside the data."""
    load_views(spark, data_dir)
    feats = spark.sql(feature_sql())
    horizon_end = F.date_add(F.col("t0"), OBS_DAYS + PRED_DAYS - 1)
    return feats.where(horizon_end <= F.lit(last_date).cast("date"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/features.parquet"))
    args = ap.parse_args()
    spark = spark_session()
    feats = build_features(spark, args.data)
    feats.toPandas().to_parquet(args.out, index=False)
    print(f"wrote {args.out}")
    spark.stop()


if __name__ == "__main__":
    main()
