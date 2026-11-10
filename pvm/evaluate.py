"""Ranking and calibration metrics for player value predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def top_share(pred: np.ndarray, actual: np.ndarray, frac: float = 0.10) -> float:
    """Share of the actual total that comes from the top `frac` of players by prediction."""
    k = max(1, int(round(len(pred) * frac)))
    idx = np.argsort(-pred, kind="stable")[:k]
    return float(actual[idx].sum() / actual.sum())


def normalized_gini(pred: np.ndarray, actual: np.ndarray) -> float:
    """Gini of actual value ordered by prediction, divided by the Gini of a perfect ordering."""
    def gini(order_by):
        a = actual[np.argsort(-order_by, kind="stable")]
        cum = np.cumsum(a) / a.sum()
        n = len(a)
        return float(cum.sum() / n - (n + 1) / (2 * n))
    return gini(pred) / gini(actual)


def value_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    return {
        "spearman": float(spearmanr(pred, actual).statistic),
        "top10_share": top_share(pred, actual, 0.10),
        "top1_share": top_share(pred, actual, 0.01),
        "norm_gini": normalized_gini(pred, actual),
        "total_pred_vs_actual": float(pred.sum() / actual.sum()),
    }


def decile_table(pred: np.ndarray, actual: np.ndarray, actual_ggr: np.ndarray,
                 pred_ggr: np.ndarray) -> pd.DataFrame:
    """Players split into 10 equal groups by predicted turnover (1 = highest)."""
    d = pd.DataFrame({"pred": pred, "turnover": actual, "pred_ggr": pred_ggr,
                      "ggr": actual_ggr})
    d["decile"] = 10 - pd.qcut(d["pred"].rank(method="first"), 10, labels=False)
    t = d.groupby("decile").agg(
        players=("pred", "size"), pred_turnover=("pred", "sum"), turnover=("turnover", "sum"),
        pred_ggr=("pred_ggr", "sum"), ggr=("ggr", "sum"))
    t["share_of_ggr"] = t["ggr"] / t["ggr"].sum()
    t["turnover_pred_vs_actual"] = t["pred_turnover"] / t["turnover"]
    t["ggr_pred_vs_actual"] = t["pred_ggr"] / t["ggr"]
    return t


def bootstrap_diff(pred_a, pred_b, actual, metric, n: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap over players of metric(pred_a) - metric(pred_b)."""
    rng = np.random.default_rng(seed)
    m = len(actual)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        diffs[i] = metric(pred_a[idx], actual[idx]) - metric(pred_b[idx], actual[idx])
    point = metric(pred_a, actual) - metric(pred_b, actual)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "ci95": [float(lo), float(hi)]}


def auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))
