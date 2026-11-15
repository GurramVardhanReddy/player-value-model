"""Retention campaign targeting: churn risk x value-if-retained, valued in NGR.

Operators value players on NGR (net gaming revenue = GGR minus bonus and promotion cost) and
spend retention budget on players who are both likely to leave and worth keeping. The bwin data
has no bonus or campaign records, so three inputs are explicit assumptions, varied in a
sensitivity grid, never presented as measured:

- bonus_rate: share of GGR given back as bonuses and promotions (NGR = GGR x (1 - bonus_rate))
- offer_cost: cost to the operator of one retention offer
- uplift: relative reduction in a targeted player's churn probability caused by the offer

What is measured on real outcomes (out-of-time cohort): churn-model discrimination and
calibration, which players each policy selects, and how many of them actually churned.
What is simulated: the incremental NGR of a campaign, which needs the uplift assumption. Real
uplift can only be measured with a randomised holdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from pvm.models import CATEGORICAL, FEATURES

POLICIES = ("random", "churn_risk", "value", "expected_incremental_ngr")


def house_margin(train: pd.DataFrame) -> float:
    """Realised house margin on the training players' next year."""
    return float(train["y_ggr"].sum() / train["y_turnover"].sum())


def _cat():
    return [FEATURES.index(c) for c in CATEGORICAL]


def churn_model(train: pd.DataFrame, seed: int = 0) -> HistGradientBoostingClassifier:
    """P(no bet in days 90-179) = 1 - P(retained)."""
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        categorical_features=_cat(), random_state=seed)
    model.fit(train[FEATURES], 1 - train["y_retained_90"])
    return model


def value_if_retained_model(train: pd.DataFrame, seed: int = 0) -> HistGradientBoostingRegressor:
    """Next-year turnover of players who stayed, so it estimates what a churner is worth if kept."""
    kept = train[train["y_retained_90"] == 1]
    model = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30,
        categorical_features=_cat(), random_state=seed)
    model.fit(kept[FEATURES], kept["y_turnover"])
    return model


def score(test: pd.DataFrame, p_churn: np.ndarray, turnover_if_kept: np.ndarray,
          margin: float, bonus_rate: float) -> pd.DataFrame:
    out = pd.DataFrame({
        "p_churn": p_churn,
        "ngr_if_retained": turnover_if_kept * margin * (1 - bonus_rate),
        "churned": 1 - test["y_retained_90"].to_numpy(int),
        "ggr_90": (test["stake_90"] - test["win_90"]).to_numpy(float),
    })
    out["value_at_risk"] = out["p_churn"] * out["ngr_if_retained"]
    return out


def select(scored: pd.DataFrame, policy: str, k: int, seed: int = 0) -> np.ndarray:
    """Indices of the k players a policy would target."""
    if policy == "random":
        return np.random.default_rng(seed).choice(len(scored), size=k, replace=False)
    key = {"churn_risk": "p_churn", "value": "ngr_if_retained",
           "expected_incremental_ngr": "value_at_risk"}[policy]
    return np.argsort(-scored[key].to_numpy(), kind="stable")[:k]


def evaluate_policy(scored: pd.DataFrame, idx: np.ndarray, uplift: float,
                    offer_cost: float) -> dict:
    """Real outcomes of the selection, plus the simulated campaign result.

    Incremental NGR counts only players who actually churned: the offer is assumed to have kept
    a share `uplift` of them, each worth their modelled NGR if retained.
    """
    s = scored.iloc[idx]
    at_risk_reached = float((s["churned"] * s["ngr_if_retained"]).sum())
    total_at_risk = float((scored["churned"] * scored["ngr_if_retained"]).sum())
    incremental = uplift * at_risk_reached - offer_cost * len(idx)
    return {
        "targeted": int(len(idx)),
        "actual_churn_rate": round(float(s["churned"].mean()), 4),
        "share_of_churned_value_reached": round(at_risk_reached / total_at_risk, 4),
        "share_of_churners_observed_ggr90_reached": round(observed_reach(scored, idx), 4),
        "share_winning_players_90d": round(float((s["ggr_90"] < 0).mean()), 4),
        "simulated_incremental_ngr": round(incremental, 2),
        "simulated_roi": round(incremental / (offer_cost * len(idx)), 3),
    }


def observed_reach(scored: pd.DataFrame, idx: np.ndarray) -> float:
    """Model-free check: share of the real first-90-day GGR of players who actually churned
    (the value the operator has seen and then lost) that falls inside the selection."""
    lost = scored["churned"] * scored["ggr_90"].clip(lower=0)
    return float(lost.iloc[idx].sum() / lost.sum())


def bootstrap_reach_diff(scored: pd.DataFrame, policy_a: str, policy_b: str, k_share: float,
                         n: int = 1000, seed: int = 0) -> dict:
    """95% CI for observed_reach(a) - observed_reach(b), resampling players and re-selecting."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        boot = scored.iloc[rng.integers(0, len(scored), len(scored))].reset_index(drop=True)
        k = int(round(k_share * len(boot)))
        diffs.append(observed_reach(boot, select(boot, policy_a, k))
                     - observed_reach(boot, select(boot, policy_b, k)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": round(float(np.mean(diffs)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    group = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return [{"bin": int(b), "mean_predicted": round(float(p[group == b].mean()), 4),
             "actual_rate": round(float(y[group == b].mean()), 4),
             "players": int((group == b).sum())}
            for b in range(bins) if (group == b).any()]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))
