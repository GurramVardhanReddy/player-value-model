"""End-to-end run: Spark features -> models -> evaluation -> results/.

Main evaluation: 5-fold cross-validation over players (out-of-fold predictions for
every player). Robustness check: out-of-time, training on players whose first stake
was in Feb-Mar 2005 and testing on players who started in Apr-Jun 2005.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pvm import evaluate as ev
from pvm import models as m
from pvm.btyd import BGNBD
from pvm.spark_features import build_features, spark_session

TEST_START = "2005-04-01"
PRODUCT_SQL = """
    SELECT UserID, SUM(sf) AS sf, SUM(wf) AS wf, SUM(sl) AS sl, SUM(wl) AS wl
    FROM (
        SELECT UserID, StakeF AS sf, WinF AS wf, 0D AS sl, 0D AS wl FROM fixed
        UNION ALL
        SELECT UserID, 0D, 0D, StakeL, WinL FROM live
    ) u
    GROUP BY UserID
"""


def margins(product: pd.DataFrame, ids) -> tuple[float, float]:
    """House margin on fixed-odds and live betting, from the given (training) players."""
    p = product[product["UserID"].isin(ids)]
    return 1 - p["wf"].sum() / p["sf"].sum(), 1 - p["wl"].sum() / p["sl"].sum()


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, product: pd.DataFrame) -> dict:
    m_fixed, m_live = margins(product, train["UserID"])
    margin = m.margin_by_product(test, m_fixed, m_live)
    bt, params = m.btyd_turnover(train, test)
    bg = BGNBD(**params["bgnbd"])
    return {
        "margin": margin,
        "run_rate": m.run_rate(test),
        "bgnbd_gamma_gamma": bt,
        "gbm_poisson": m.gbm_turnover(train, test),
        "gbm_ggr_direct": m.gbm_ggr_direct(train, test),
        "ret_gbm": m.gbm_retention(train, test),
        "ret_p_alive": bg.p_alive(test["active_days_90"] - 1, test["last_active_day"], 90),
        "btyd_days": bg.expected_days(365, test["active_days_90"] - 1,
                                      test["last_active_day"], 90),
        "btyd_params": params,
        "margins": (m_fixed, m_live),
    }


def report(test: pd.DataFrame, p: dict, boot: bool = True) -> dict:
    y_to = test["y_turnover"].to_numpy(float)
    y_ggr = test["y_ggr"].to_numpy(float)
    y_ret = test["y_retained_90"].to_numpy(int)
    turnover_models = ("run_rate", "bgnbd_gamma_gamma", "gbm_poisson")
    out = {
        "players": int(len(test)),
        "turnover": {k: ev.value_metrics(p[k], y_to) for k in turnover_models},
        "ggr": {k: ev.value_metrics(p[k] * p["margin"], y_ggr) for k in turnover_models},
        "retention_90d": {
            "base_rate": float(y_ret.mean()),
            "auc_gbm": ev.auc(y_ret, p["ret_gbm"]),
            "auc_recency_only": ev.auc(y_ret, -test["recency"].to_numpy()),
            "auc_bgnbd_p_alive": ev.auc(y_ret, p["ret_p_alive"]),
        },
        "bgnbd_active_days_pred_vs_actual": float(p["btyd_days"].sum()
                                                  / test["y_active_days"].sum()),
    }
    out["ggr"]["gbm_ggr_direct"] = ev.value_metrics(p["gbm_ggr_direct"], y_ggr)
    if boot:
        g = p["gbm_poisson"] * p["margin"]
        out["bootstrap_ggr_top10_share"] = {
            "gbm_minus_bgnbd": ev.bootstrap_diff(
                g, p["bgnbd_gamma_gamma"] * p["margin"], y_ggr, ev.top_share),
            "gbm_minus_run_rate": ev.bootstrap_diff(
                g, p["run_rate"] * p["margin"], y_ggr, ev.top_share),
            "turnover_x_margin_minus_direct_ggr": ev.bootstrap_diff(
                g, p["gbm_ggr_direct"], y_ggr, ev.top_share),
        }
        out["bootstrap_turnover_gini"] = {
            "gbm_minus_run_rate": ev.bootstrap_diff(
                p["gbm_poisson"], p["run_rate"], y_to, ev.normalized_gini),
            "gbm_minus_bgnbd": ev.bootstrap_diff(
                p["gbm_poisson"], p["bgnbd_gamma_gamma"], y_to, ev.normalized_gini),
        }
    return out


def gains_chart(actual: np.ndarray, preds: dict, path: Path) -> None:
    """Share of next-year GGR captured when targeting the top x% of players."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(actual)
    xs = np.arange(1, n + 1) / n
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    ax.plot(xs, np.cumsum(np.sort(actual)[::-1]) / actual.sum(), color="#9aa0a6",
            lw=1.5, ls="--", label="Perfect foresight")
    for (name, pred), color in zip(preds.items(), ("#1f6feb", "#d97706", "#7c3aed"),
                                   strict=True):
        ax.plot(xs, np.cumsum(actual[np.argsort(-pred, kind="stable")]) / actual.sum(),
                color=color, lw=2, label=name)
    ax.axvline(0.10, color="#d0d7de", lw=1)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Share of players targeted (ranked by predicted value)")
    ax.set_ylabel("Share of next-year GGR captured")
    ax.set_title("Out-of-fold gains, 41,247 bwin players (5-fold CV)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spark = spark_session()
    feats = build_features(spark, args.data).toPandas()
    product = spark.sql(PRODUCT_SQL).toPandas()
    spark.stop()

    feats["t0"] = pd.to_datetime(feats["t0"])
    df = m.add_features(feats).sort_values("UserID").reset_index(drop=True)

    # 1) 5-fold cross-validation: out-of-fold predictions for every player
    keys = ("margin", "run_rate", "bgnbd_gamma_gamma", "gbm_poisson", "gbm_ggr_direct",
            "ret_gbm", "ret_p_alive", "btyd_days")
    oof = {k: np.zeros(len(df)) for k in keys}
    fold_params = []
    for tr_idx, te_idx in KFold(args.folds, shuffle=True, random_state=42).split(df):
        p = fit_predict(df.iloc[tr_idx].reset_index(drop=True),
                        df.iloc[te_idx].reset_index(drop=True), product)
        for k in keys:
            oof[k][te_idx] = p[k]
        fold_params.append(p["btyd_params"])
    m_fixed, m_live = margins(product, df["UserID"])

    results = {
        "data": {
            "players": int(len(df)),
            "first_stake_range": [str(df["t0"].min().date()), str(df["t0"].max().date())],
            "share_active_next_year": float((df["y_active_days"] > 0).mean()),
            "turnover_next_year": float(df["y_turnover"].sum()),
            "ggr_next_year": float(df["y_ggr"].sum()),
            "share_of_ggr_from_top10pct_players_actual": ev.top_share(
                df["y_ggr"].to_numpy(float), df["y_ggr"].to_numpy(float)),
            "house_margin_fixed_odds": m_fixed, "house_margin_live": m_live,
        },
        "cv5": report(df, oof),
        "btyd_params_fold1": fold_params[0],
    }
    both = df[(df["y_turnover"] > 0) & (df["stake_90"] > 0)]
    results["payout_ratio_persistence_spearman"] = float(
        both["payout_ratio_90"].corr(1 - both["y_ggr"] / both["y_turnover"], method="spearman"))
    results["stake_level_persistence_spearman"] = float(
        both["stake_90"].corr(both["y_turnover"], method="spearman"))

    # 2) Out-of-time check: later cohort
    train = df[df["t0"] < TEST_START].reset_index(drop=True)
    test = df[df["t0"] >= TEST_START].reset_index(drop=True)
    results["out_of_time"] = report(test, fit_predict(train, test, product), boot=False)
    results["out_of_time"]["train_players"] = int(len(train))

    dec = ev.decile_table(oof["gbm_poisson"], df["y_turnover"].to_numpy(float),
                          df["y_ggr"].to_numpy(float),
                          pred_ggr=oof["gbm_poisson"] * oof["margin"])
    dec.to_csv(args.out / "deciles_gbm_cv5.csv")
    gains_chart(df["y_ggr"].to_numpy(float), {
        "Gradient boosting (turnover x margin)": oof["gbm_poisson"] * oof["margin"],
        "Run-rate (last 30 days x 12)": oof["run_rate"] * oof["margin"],
        "BG/NBD + Gamma-Gamma": oof["bgnbd_gamma_gamma"] * oof["margin"],
    }, args.out / "gains_ggr.png")
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(dec.round(3).to_string())


if __name__ == "__main__":
    main()
