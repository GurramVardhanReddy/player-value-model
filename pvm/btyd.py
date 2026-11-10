"""BG/NBD and Gamma-Gamma lifetime value models, fitted by maximum likelihood.

BG/NBD (Fader, Hardie and Lee, 2005) models how many days a player bets on and when
they stop. Gamma-Gamma (Fader, Hardie and Lee, 2005b) models the stake per betting
day. Together they give expected turnover over a future period.

Inputs per player, in days since the first stake:
  x   repeat active days (active days in the window minus the first)
  t_x day of the last active day
  T   length of the observation window
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1


@dataclass
class BGNBD:
    r: float = 1.0
    alpha: float = 1.0
    a: float = 1.0
    b: float = 1.0

    @staticmethod
    def _neg_ll(log_params, x, t_x, T):
        r, alpha, a, b = np.exp(log_params)
        common = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        a1 = betaln(a, b + x) - betaln(a, b) + common - (r + x) * np.log(alpha + T)
        with np.errstate(divide="ignore", invalid="ignore"):
            a2 = (
                betaln(a + 1, b + x - 1) - betaln(a, b) + common
                - (r + x) * np.log(alpha + t_x)
            )
        a2 = np.where(x > 0, a2, -np.inf)
        ll = np.logaddexp(a1, a2)
        return -np.sum(ll)

    def fit(self, x, t_x, T) -> BGNBD:
        x, t_x, T = (np.asarray(v, dtype=float) for v in (x, t_x, T))
        res = minimize(
            self._neg_ll, np.zeros(4), args=(x, t_x, T), method="L-BFGS-B",
            bounds=[(-10, 10)] * 4,
        )
        self.r, self.alpha, self.a, self.b = np.exp(res.x)
        self.converged = bool(res.success)
        return self

    def expected_days(self, t, x, t_x, T):
        """Expected active days in (T, T + t] given the player's history."""
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        x, t_x, T = (np.asarray(v, dtype=float) for v in (x, t_x, T))
        z = t / (alpha + T + t)
        hyp = hyp2f1(r + x, b + x, a + b + x - 1, z)
        num = (a + b + x - 1) / (a - 1) * (
            1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            den_term = a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x)
        den = 1 + np.where(x > 0, den_term, 0.0)
        return num / den

    def p_alive(self, x, t_x, T):
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        x, t_x, T = (np.asarray(v, dtype=float) for v in (x, t_x, T))
        with np.errstate(divide="ignore", invalid="ignore"):
            term = a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x)
        return np.where(x > 0, 1 / (1 + term), 1.0)


@dataclass
class GammaGamma:
    p: float = 1.0
    q: float = 1.0
    v: float = 1.0

    @staticmethod
    def _neg_ll(log_params, n, m):
        p, q, v = np.exp(log_params)
        ll = (
            gammaln(p * n + q) - gammaln(p * n) - gammaln(q) + q * np.log(v)
            + (p * n - 1) * np.log(m) + p * n * np.log(n) - (p * n + q) * np.log(n * m + v)
        )
        return -np.sum(ll)

    def fit(self, n, m) -> GammaGamma:
        """n = active days, m = mean stake per active day (players with n >= 1, m > 0)."""
        n, m = np.asarray(n, dtype=float), np.asarray(m, dtype=float)
        res = minimize(
            self._neg_ll, np.zeros(3), args=(n, m), method="L-BFGS-B",
            bounds=[(-10, 12)] * 3,
        )
        self.p, self.q, self.v = np.exp(res.x)
        self.converged = bool(res.success)
        return self

    def expected_stake(self, n, m):
        """Expected stake per active day, shrunk towards the population mean."""
        n, m = np.asarray(n, dtype=float), np.asarray(m, dtype=float)
        return self.p * (self.v + n * m) / (self.p * n + self.q - 1)
