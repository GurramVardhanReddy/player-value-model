import numpy as np

from pvm import evaluate as ev
from pvm.btyd import BGNBD, GammaGamma


def simulate_bgnbd(n, r, alpha, a, b, T, seed=0):
    """Simulate BG/NBD histories in continuous time."""
    rng = np.random.default_rng(seed)
    lam = rng.gamma(r, 1 / alpha, n)
    p = rng.beta(a, b, n)
    x = np.zeros(n)
    t_x = np.zeros(n)
    for i in range(n):
        t = 0.0
        while True:
            t += rng.exponential(1 / lam[i])
            if t > T:
                break
            x[i] += 1
            t_x[i] = t
            if rng.random() < p[i]:
                break
    return x, t_x, np.full(n, T)


def test_bgnbd_recovers_parameters():
    x, t_x, T = simulate_bgnbd(4000, r=0.8, alpha=4.0, a=1.2, b=6.0, T=90.0)
    bg = BGNBD().fit(x, t_x, T)
    assert bg.converged
    assert abs(bg.r - 0.8) / 0.8 < 0.2
    assert abs(bg.alpha - 4.0) / 4.0 < 0.3


def test_bgnbd_expectations_are_sane():
    bg = BGNBD(r=1.0, alpha=3.0, a=1.3, b=20.0)
    # more past activity and a recent last bet -> more expected future days
    lo = bg.expected_days(365, np.array([2.0]), np.array([10.0]), np.array([90.0]))
    hi = bg.expected_days(365, np.array([40.0]), np.array([89.0]), np.array([90.0]))
    assert 0 < lo[0] < hi[0]
    # a frequent bettor who went quiet long ago is less likely to be alive
    pa_recent = bg.p_alive(np.array([30.0]), np.array([89.0]), np.array([90.0]))
    pa_stale = bg.p_alive(np.array([30.0]), np.array([20.0]), np.array([90.0]))
    assert pa_recent[0] > pa_stale[0]
    assert bg.p_alive(np.array([0.0]), np.array([0.0]), np.array([90.0]))[0] == 1.0


def test_gamma_gamma_shrinks_towards_population_mean():
    rng = np.random.default_rng(1)
    n = rng.integers(1, 30, 3000).astype(float)
    nu = rng.gamma(3.0, 1 / 0.05, 3000)  # per-player mean stake, population mean 60
    m = np.array([rng.gamma(2.0, 1 / nu_i, int(k)).mean()
                  for k, nu_i in zip(n.astype(int), nu, strict=True)])
    gg = GammaGamma().fit(n, m)
    one_day_big = gg.expected_stake(np.array([1.0]), np.array([1000.0]))[0]
    many_days_big = gg.expected_stake(np.array([50.0]), np.array([1000.0]))[0]
    assert one_day_big < many_days_big < 1000.0


def test_top_share_and_gini():
    actual = np.array([100.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert ev.top_share(actual, actual, 0.10) == 1.0
    assert ev.normalized_gini(actual, actual) == 1.0
    reverse = -actual
    assert ev.top_share(reverse, actual, 0.10) == 0.0
    assert ev.normalized_gini(reverse, actual) < 0


def test_bootstrap_diff_zero_for_identical_models():
    rng = np.random.default_rng(0)
    actual = rng.gamma(0.5, 100, 500)
    pred = actual + rng.normal(0, 10, 500)
    out = ev.bootstrap_diff(pred, pred, actual, ev.top_share, n=50)
    assert out["diff"] == 0.0 and out["ci95"] == [0.0, 0.0]
