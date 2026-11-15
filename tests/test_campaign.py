import numpy as np
import pandas as pd

from pvm import campaign as c


def scored_frame():
    # 6 players: churn flag, modelled NGR if kept, churn probability, observed 90-day GGR
    return pd.DataFrame({
        "p_churn":         [0.9, 0.8, 0.2, 0.1, 0.7, 0.3],
        "ngr_if_retained": [10., 5., 500., 400., 300., 1.],
        "churned":         [1, 1, 0, 0, 1, 0],
        "ggr_90":          [5., 2., 200., 150., 100., -50.],
    }).assign(value_at_risk=lambda d: d.p_churn * d.ngr_if_retained)


def test_policies_pick_expected_players():
    s = scored_frame()
    assert set(c.select(s, "churn_risk", 2)) == {0, 1}
    assert set(c.select(s, "value", 2)) == {2, 3}
    assert set(c.select(s, "expected_incremental_ngr", 2)) == {4, 2}   # 210 and 100
    assert len(set(c.select(s, "random", 3, seed=1))) == 3


def test_evaluate_counts_only_actual_churners():
    s = scored_frame()
    r = c.evaluate_policy(s, np.array([4, 2]), uplift=0.1, offer_cost=10)
    # only player 4 churned: 0.1 * 300 - 2 * 10 = 10
    assert r["simulated_incremental_ngr"] == 10.0
    assert r["actual_churn_rate"] == 0.5
    assert abs(r["share_of_churned_value_reached"] - 300 / 315) < 1e-3
    assert abs(r["share_of_churners_observed_ggr90_reached"] - 100 / 107) < 1e-3


def test_calibration_and_brier():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(c.brier(y, p) - 0.025) < 1e-9
    rows = c.calibration_table(y, p, bins=2)
    assert sum(r["players"] for r in rows) == 4
