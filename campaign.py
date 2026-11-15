"""Retention campaign study on the out-of-time cohort.

Train on players whose first stake was in Feb-Mar 2005, test on players who started in
Apr-Jun 2005 (the same split as run.py's out-of-time check). Reads the Spark feature table
that run.py writes to results/features.parquet.

    python campaign.py --features results/features.parquet
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd

from pvm import campaign as c
from pvm import evaluate as ev
from pvm.models import add_features

TEST_START = "2005-04-01"
BUDGET_SHARE = 0.10          # target 10% of the cohort
BASE = {"bonus_rate": 0.20, "offer_cost": 10.0, "uplift": 0.10}   # assumptions, see pvm/campaign.py
GRID = {"bonus_rate": (0.10, 0.20, 0.30), "offer_cost": (5.0, 10.0, 20.0),
        "uplift": (0.05, 0.10, 0.20)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("results/features.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/campaign.json"))
    args = ap.parse_args()

    df = add_features(pd.read_parquet(args.features))
    df["t0"] = pd.to_datetime(df["t0"])
    train = df[df["t0"] < TEST_START].reset_index(drop=True)
    test = df[df["t0"] >= TEST_START].reset_index(drop=True)

    margin = c.house_margin(train)
    p_churn = c.churn_model(train).predict_proba(test[c.FEATURES])[:, 1]
    turnover_if_kept = c.value_if_retained_model(train).predict(test[c.FEATURES])
    y_churn = 1 - test["y_retained_90"].to_numpy(int)
    k = int(round(BUDGET_SHARE * len(test)))

    def run(bonus_rate, offer_cost, uplift):
        scored = c.score(test, p_churn, turnover_if_kept, margin, bonus_rate)
        return {pol: c.evaluate_policy(scored, c.select(scored, pol, k), uplift, offer_cost)
                for pol in c.POLICIES}

    base = run(**BASE)
    scored_base = c.score(test, p_churn, turnover_if_kept, margin, BASE["bonus_rate"])
    boot = {f"expected_incremental_ngr_minus_{b}": c.bootstrap_reach_diff(
        scored_base, "expected_incremental_ngr", b, BUDGET_SHARE) for b in ("churn_risk", "value")}
    grid = []
    for b, o, u in product(GRID["bonus_rate"], GRID["offer_cost"], GRID["uplift"]):
        r = run(b, o, u)
        best = max(r, key=lambda pol: r[pol]["simulated_incremental_ngr"])
        grid.append({"bonus_rate": b, "offer_cost": o, "uplift": u, "best_policy": best,
                     **{f"{pol}_incremental_ngr": r[pol]["simulated_incremental_ngr"]
                        for pol in c.POLICIES}})

    out = {
        "split": {"train_players": int(len(train)), "test_players": int(len(test)),
                  "test_first_stake_from": TEST_START},
        "house_margin_train": round(margin, 4),
        "churn_model": {
            "test_churn_rate": round(float(y_churn.mean()), 4),
            "auc": round(ev.auc(y_churn, p_churn), 4),
            "brier": round(c.brier(y_churn, p_churn), 4),
            "brier_base_rate": round(c.brier(y_churn, y_churn.mean() + 0 * p_churn), 4),
            "calibration_deciles": c.calibration_table(y_churn, p_churn),
        },
        "assumptions_base": BASE,
        "budget": {"share": BUDGET_SHARE, "players_targeted": k},
        "policies_base": base,
        "bootstrap_observed_reach_diff": boot,
        "sensitivity": grid,
        "sensitivity_best_policy_counts": pd.Series([g["best_policy"] for g in grid])
        .value_counts().to_dict(),
    }
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    keys = ("split", "churn_model", "policies_base", "bootstrap_observed_reach_diff",
            "sensitivity_best_policy_counts")
    print(json.dumps({k_: out[k_] for k_ in keys}, indent=2))


if __name__ == "__main__":
    main()
