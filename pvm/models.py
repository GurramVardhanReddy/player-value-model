"""Player value models: run-rate baseline, BG/NBD + Gamma-Gamma, gradient boosting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from pvm.btyd import BGNBD, GammaGamma
from pvm.spark_features import OBS_DAYS, PRED_DAYS

TOP_COUNTRIES = (276, 792, 616, 724, 300, 40, 380)  # coded as-is, the rest grouped


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived ratios and encodings on top of the Spark SQL feature table."""
    d = df.copy()
    eps = 1e-9
    d["age_at_first_bet"] = pd.to_datetime(d["t0"]).dt.year - d["BirthYear"]
    d["live_share"] = d["stake_live_90"] / (d["stake_90"] + eps)
    d["stake_per_active_day"] = d["stake_90"] / d["active_days_90"].clip(lower=1)
    d["stake_per_bet"] = d["stake_90"] / d["bets_90"].clip(lower=1)
    d["payout_ratio_90"] = d["win_90"] / (d["stake_90"] + eps)
    d["recency"] = OBS_DAYS - 1 - d["last_active_day"]
    d["m3_vs_m1"] = np.log1p(d["stake_m3"]) - np.log1p(d["stake_m1"])
    d["peak_day_share"] = d["max_day_stake"] / (d["stake_90"] + eps)
    d["country"] = np.where(d["CountryID"].isin(TOP_COUNTRIES), d["CountryID"], 0)
    for c in ("stake_90", "stake_wk1", "stake_m1", "stake_m3", "max_day_stake",
              "max_7d_stake", "stake_per_active_day", "stake_per_bet"):
        d[f"log_{c}"] = np.log1p(d[c])
    return d


FEATURES = [
    "log_stake_90", "log_stake_wk1", "log_stake_m1", "log_stake_m3", "log_max_day_stake",
    "log_max_7d_stake", "log_stake_per_active_day", "log_stake_per_bet",
    "active_days_90", "active_days_m3", "active_days_last14", "bets_90", "live_days_90",
    "live_share", "payout_ratio_90", "recency", "m3_vs_m1", "peak_day_share",
    "mean_gap", "max_gap", "sd_gap", "half_stake_day",
    "reg_to_first_bet", "reg_to_first_dep", "age_at_first_bet", "Gender", "country",
]
CATEGORICAL = ["country", "Gender"]


def run_rate(df: pd.DataFrame) -> np.ndarray:
    """Baseline: last 30 days of stake, scaled up to a year."""
    return df["stake_m3"].to_numpy() * PRED_DAYS / 30


def btyd_turnover(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    def inputs(d):
        n = d["active_days_90"].to_numpy(float)
        return n - 1, d["last_active_day"].to_numpy(float), np.full(len(d), OBS_DAYS, float)

    bg = BGNBD().fit(*inputs(train))
    m_tr = train["stake_per_active_day"].to_numpy(float)
    ok = m_tr > 0
    gg = GammaGamma().fit(train["active_days_90"].to_numpy(float)[ok], m_tr[ok])
    x, t_x, T = inputs(test)
    days = bg.expected_days(PRED_DAYS, x, t_x, T)
    stake = gg.expected_stake(test["active_days_90"].to_numpy(float),
                              test["stake_per_active_day"].to_numpy(float))
    params = {"bgnbd": {k: round(float(getattr(bg, k)), 4) for k in ("r", "alpha", "a", "b")},
              "gamma_gamma": {k: round(float(getattr(gg, k)), 4) for k in ("p", "q", "v")},
              "converged": bool(bg.converged and gg.converged)}
    return days * stake, params


def gbm_turnover(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Gradient boosting with a Poisson loss on next-year turnover (non-negative, heavy-tailed)."""
    model = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL], random_state=seed,
    )
    model.fit(train[FEATURES], train["y_turnover"])
    return model.predict(test[FEATURES])


def gbm_ggr_direct(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Comparison: regress realised GGR directly (it can be negative, so squared error)."""
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL], random_state=seed,
    )
    model.fit(train[FEATURES], train["y_ggr"])
    return model.predict(test[FEATURES])


def gbm_retention(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """P(player bets at least once in days 90-179)."""
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL], random_state=seed,
    )
    model.fit(train[FEATURES], train["y_retained_90"])
    return model.predict_proba(test[FEATURES])[:, 1]


def margin_by_product(df: pd.DataFrame, fixed_margin: float, live_margin: float) -> np.ndarray:
    """Expected house margin per player from their fixed-odds / live mix."""
    share = df["live_share"].to_numpy()
    return share * live_margin + (1 - share) * fixed_margin
