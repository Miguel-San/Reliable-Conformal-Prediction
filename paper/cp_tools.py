"""Helpers for the figures of the paper: synthetic data, score laws and Monte Carlo routines.

The order-statistic indices, the achieved confidence and the guaranteed minimum coverage come from the package
``reliable_conformal_prediction``; this module holds only what the package does not provide.

Conventions
-----------
N      calibration-set size
m      1-based index of the order statistic used as threshold; s_(m) is the m-th smallest calibration score
C      realised (calibration-conditional) coverage, C = F_S(s_(m)) ~ Beta(m, N + 1 - m) for iid continuous scores
m == N + 1 encodes "no admissible order statistic": the threshold is +inf (trivial interval, coverage 1).

Thresholds are always selected by integer index (np.partition), never through np.quantile.
"""
import zlib
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np
import scipy.stats as st
from scipy import special

BASE_SEED = 2026


def _out(a):
    """Return a Python scalar for 0-d results and the array otherwise."""
    a = np.asarray(a)
    return a.item() if a.ndim == 0 else a


# ----------------------------------------------------------------------------------------------------------
# A. Thresholds and laws
# ----------------------------------------------------------------------------------------------------------
def order_stats(scores, ms):
    """Thresholds s_(m) along the last axis for each named index in `ms` (m == N + 1 gives +inf)."""
    s = np.asarray(scores, float)
    N = s.shape[-1]
    kth = sorted({int(m) - 1 for m in ms.values() if m <= N})
    part = np.partition(s, kth, axis=-1) if kth else s
    return {k: (part[..., int(m) - 1] if m <= N else np.full(s.shape[:-1], np.inf)) for k, m in ms.items()}


def coverage_law(N, m):
    """Law of the realised coverage C for the threshold s_(m): Beta(m, N + 1 - m) (requires m <= N)."""
    return st.beta(m, N + 1 - m)


def test_count_law(n_test, N, m):
    """Covered points in an independent test set of size n_test: BetaBinomial(n_test, m, N + 1 - m).

    The same law holds exactly for random calibration/test splits of any fixed pool of distinct scores.
    """
    return st.betabinom(n_test, m, N + 1 - m)


def threshold_quantile(law, N, m, p):
    """Exact p-quantile of the threshold s_(m): F_S^{-1}(Beta^{-1}(p; m, N + 1 - m)); +inf if m == N + 1.

    Evaluated through the upper tail (isf) for accuracy under heavy tails. Vectorised over N, m and p.
    """
    N, m, p = np.broadcast_arrays(np.asarray(N), np.asarray(m), np.asarray(p, float))
    feasible = m <= N
    mm, NN = np.where(feasible, m, 1), np.where(feasible, N, 1)
    upper = st.beta.ppf(1 - p, NN + 1 - mm, mm)  # = 1 - Beta^{-1}(p; m, N + 1 - m)
    return _out(np.where(feasible, law.isf(upper), np.inf))


# ----------------------------------------------------------------------------------------------------------
# B. Noise and score laws
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Noise:
    """Symmetric noise eps = scale * xi, with xi from a standard family ('t' needs nu)."""

    family: str = "gaussian"
    nu: Optional[float] = None
    scale: float = 1.0

    @property
    def base(self):
        return {"uniform": lambda: st.uniform(loc=-1, scale=2), "gaussian": st.norm, "laplace": st.laplace,
                "t": lambda: st.t(self.nu), "cauchy": st.cauchy}[self.family]()

    def _std_cdf(self, z):
        """CDF of the standard family through scipy.special (fast and accurate in both tails)."""
        z = np.asarray(z, float)
        if self.family == "gaussian":
            return special.ndtr(z)
        if self.family == "t":
            return special.stdtr(self.nu, z)
        if self.family == "cauchy":
            return np.where(z < 0, np.arctan2(1.0, -z) / np.pi, 1 - np.arctan2(1.0, z) / np.pi)
        if self.family == "laplace":
            return np.where(z < 0, 0.5 * np.exp(np.minimum(z, 0)), 1 - 0.5 * np.exp(-np.maximum(z, 0)))
        return np.clip((z + 1) / 2, 0.0, 1.0)  # uniform on (-1, 1)

    def cdf(self, e):
        return self._std_cdf(np.asarray(e, float) / self.scale)

    def sf(self, e):
        return self._std_cdf(-np.asarray(e, float) / self.scale)  # symmetric noise

    def rvs(self, rng, size):
        xi = {"uniform": lambda: rng.uniform(-1.0, 1.0, size), "gaussian": lambda: rng.standard_normal(size),
              "laplace": lambda: rng.laplace(0.0, 1.0, size), "t": lambda: rng.standard_t(self.nu, size),
              "cauchy": lambda: rng.standard_cauchy(size)}[self.family]()
        return self.scale * xi

    def abs_ppf(self, u):
        """Quantile function of |eps|."""
        return self.scale * self.base.ppf((1 + np.asarray(u, float)) / 2)

    def with_scale(self, scale):
        return replace(self, scale=scale)


def oracle_scaled(family, nu=None, c_ref=0.9, target=1.0):
    """Noise whose absolute value has its c_ref-quantile at `target` (true threshold t* = target)."""
    noise = Noise(family, nu)
    return noise.with_scale(target / float(noise.abs_ppf(c_ref)))


FAMILY_LABELS = {"uniform": "Uniform", "gaussian": "Gaussian", "laplace": "Laplace", "t3": "Student-t (ν=3)",
                 "cauchy": "Cauchy"}


def noise_zoo(c_ref=0.9):
    """Score families ordered from light to heavy tails, all scaled so that F_S^{-1}(c_ref) = 1."""
    return {"uniform": oracle_scaled("uniform", c_ref=c_ref), "gaussian": oracle_scaled("gaussian", c_ref=c_ref),
            "laplace": oracle_scaled("laplace", c_ref=c_ref), "t3": oracle_scaled("t", 3, c_ref=c_ref),
            "cauchy": oracle_scaled("cauchy", c_ref=c_ref)}


class FoldedScoreLaw:
    """Law of the score S = |eps| (homoscedastic noise and fhat = f)."""

    def __init__(self, noise):
        self.noise = noise

    def cdf(self, s):
        s = np.asarray(s, float)
        return np.where(s > 0, 1 - 2 * self.noise.sf(np.abs(s)), 0.0)

    def ppf(self, u):
        return self.noise.abs_ppf(u)

    def isf(self, p):
        return self.noise.scale * self.noise.base.isf(np.asarray(p, float) / 2)

    def rvs(self, rng, size):
        return np.abs(self.noise.rvs(rng, size))


def gauss_legendre(a, b, n_panels=64, order=8, breaks=()):
    """Composite Gauss-Legendre nodes and weights on [a, b], with panel edges at the given break points."""
    edges = np.unique(np.concatenate([np.linspace(a, b, n_panels + 1), [x for x in breaks if a < x < b]]))
    t, w = np.polynomial.legendre.leggauss(order)
    lo, hi = edges[:-1, None], edges[1:, None]
    return ((lo + hi) / 2 + (hi - lo) / 2 * t).ravel(), ((hi - lo) / 2 * w).ravel()


def _bisect(fun, target, lo, hi, n_iter=200):
    """Vectorised bisection for an increasing function: fun(t) = target on [lo, hi] (hi expanded if needed)."""
    target = np.asarray(target, float)
    lo = np.broadcast_to(np.asarray(lo, float), target.shape).copy()
    hi = np.broadcast_to(np.asarray(hi, float), target.shape).copy()
    for _ in range(200):
        short = fun(hi) < target
        if not short.any():
            break
        hi = np.where(short, 2 * hi, hi)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        below = fun(mid) < target
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        if np.all(hi - lo <= 1e-13 * np.maximum(hi, 1e-300)):
            break
    return (lo + hi) / 2


class MixtureScoreLaw:
    """Law of the score S = |y - fhat(x)| when Y = f(X) + sigma(X) eps, by quadrature over x.

    With bias b(x) = f(x) - fhat(x), the band fhat(x) +- t covers x with probability
    c(x; t) = F_eps((t - b) / sigma) - F_eps((-t - b) / sigma), and F_S(t) = int c(x; t) p(x) dx.
    """

    def __init__(self, noise, x, w, bias, sigma):
        self.noise, self.x = noise, np.asarray(x, float)
        self.w = np.asarray(w, float) / np.sum(w)
        self.b, self.sigma = np.asarray(bias, float), np.asarray(sigma, float)
        self._table = None

    def _integrate(self, t, fun, chunk=2048):
        t = np.asarray(t, float)
        flat, out = t.ravel(), np.empty(t.size)
        for i in range(0, flat.size, chunk):
            out[i:i + chunk] = fun(flat[i:i + chunk, None]) @ self.w
        return out.reshape(t.shape)

    def cdf_direct(self, t):
        """F_S(t) by quadrature for every t (exact up to the quadrature error)."""
        b, s = self.b, self.sigma
        return np.clip(self._integrate(t, lambda u: self.noise.cdf((u - b) / s) - self.noise.cdf((-u - b) / s)),
                       0.0, 1.0)

    def cdf(self, t, n_grid=30_000, tail=1e-7):
        """F_S(t). Large inputs (Monte Carlo thresholds) use a table of cdf_direct on a grid uniform in log(1 + t),
        linearly interpolated (error ~1e-7); thresholds beyond the table are evaluated directly."""
        t = np.asarray(t, float)
        if t.size < 5_000:
            return self.cdf_direct(t)
        if self._table is None:
            t_hi = float(self.isf(tail))
            z = np.linspace(0.0, np.log1p(t_hi), n_grid)
            self._table = (z, self.cdf_direct(np.expm1(z)), t_hi)
        z, F, t_hi = self._table
        out = np.interp(np.log1p(np.clip(t, 0.0, None)), z, F)
        beyond = ~(t <= t_hi)
        if beyond.any():
            out[beyond] = self.cdf_direct(t[beyond])
        return out

    def sf(self, t):
        b, s = self.b, self.sigma
        return np.clip(self._integrate(t, lambda u: self.noise.sf((u - b) / s) + self.noise.cdf((-u - b) / s)),
                       0.0, 1.0)

    def ppf(self, u):
        return _bisect(self.cdf, u, 0.0, 1.0)

    def isf(self, p):
        return _bisect(lambda t: -self.sf(t), -np.asarray(p, float), 0.0, 1.0)


# ----------------------------------------------------------------------------------------------------------
# C. Synthetic data-generating process
# ----------------------------------------------------------------------------------------------------------
def f_poly(x):
    """Smooth 1-D test function (the one used in the package's MAPIE example)."""
    x = np.asarray(x, float)
    return 5 * x + 5 * x ** 4 - 9 * x ** 2


def sigma_const(x):
    return np.ones_like(np.asarray(x, float))


def two_regime(edge=0.7, ratio=3.0):
    """Noise multiplier 1 for x < edge (benign regime) and `ratio` for x >= edge (harsh regime)."""
    return lambda x: np.where(np.asarray(x, float) < edge, 1.0, ratio)


@dataclass(frozen=True)
class SyntheticDGP:
    """Y = f(X) + sigma(X) eps with X ~ U(0, 1); `breaks` lists discontinuities of sigma (for quadrature)."""

    noise: Noise = field(default_factory=Noise)
    f: Callable = f_poly
    sigma: Callable = sigma_const
    breaks: tuple = ()

    def sample(self, rng, size):
        x = rng.uniform(0.0, 1.0, size)
        return x, self.f(x) + self.sigma(x) * self.noise.rvs(rng, size)

    def score(self, x, y, fhat=None):
        return np.abs(y - (self.f(x) if fhat is None else fhat(x)))

    def cond_coverage(self, x, t):
        """Coverage at input x of the band f(x) +- t."""
        s = self.sigma(np.asarray(x, float))
        return self.noise.cdf(t / s) - self.noise.cdf(-t / s)

    def cond_oracle(self, x, c):
        """Half-width of the narrowest band around f with coverage c at x (symmetric noise)."""
        return self.sigma(np.asarray(x, float)) * self.noise.abs_ppf(c)

    def score_law(self, fhat=None, n_panels=64):
        """Marginal law of the score, by quadrature over X ~ U(0, 1)."""
        x, w = gauss_legendre(0.0, 1.0, n_panels, breaks=self.breaks)
        bias = np.zeros_like(x) if fhat is None else self.f(x) - fhat(x)
        return MixtureScoreLaw(self.noise, x, w, bias, self.sigma(x))


def calibrate_scale(dgp, c_ref=0.9, target=1.0):
    """Copy of dgp with the noise rescaled so that the marginal score quantile at c_ref is `target` (fhat = f)."""
    q = float(dgp.score_law().ppf(c_ref))
    return replace(dgp, noise=dgp.noise.with_scale(dgp.noise.scale * target / q))


def frozen(model, n_grid=4001):
    """Freeze a fitted 1-D regressor as a piecewise-linear function on a fine grid of [0, 1].

    The tabulated function *is* the frozen model used everywhere (sampling and quadrature), so exactness is
    preserved, and evaluation is cheap.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    values = model.predict(grid[:, None])
    return lambda x: np.interp(x, grid, values)


# ----------------------------------------------------------------------------------------------------------
# D. Monte Carlo
# ----------------------------------------------------------------------------------------------------------
def seed_for(key, base=BASE_SEED):
    """Fresh SeedSequence for an experiment key (reproducible, independent of the order in which cells run)."""
    return np.random.SeedSequence([base, zlib.crc32(str(key).encode())])


def _rngs(key, R, chunk):
    children = seed_for(key).spawn(-(-R // chunk))
    return [(np.random.default_rng(c), min(chunk, R - i * chunk)) for i, c in enumerate(children)]


def run_mc(dgp, N, ms, R, key, law, fhat=None, chunk=20_000):
    """R calibration sets of size N drawn through the regression pipeline (common random numbers).

    Returns, per named index in `ms`, the thresholds and the exact realised coverage law.cdf(threshold).
    """
    thr = {k: [] for k in ms}
    for rng, r in _rngs(key, R, chunk):
        x, y = dgp.sample(rng, (r, N))
        t = order_stats(dgp.score(x, y, fhat), ms)
        for k in ms:
            thr[k].append(t[k])
    out = {"thr": {k: np.concatenate(v) for k, v in thr.items()}}
    out["cov"] = {k: law.cdf(v) for k, v in out["thr"].items()}
    return out


def score_mc(law, N, ms, R, key, chunk=50_000):
    """Score-level Monte Carlo: thresholds from R iid score samples of size N."""
    thr = {k: [] for k in ms}
    for rng, r in _rngs(key, R, chunk):
        t = order_stats(law.rvs(rng, (r, N)), ms)
        for k in ms:
            thr[k].append(t[k])
    return {k: np.concatenate(v) for k, v in thr.items()}


def simulate_splits(pool, N, ms, R, key, chunk=20_000):
    """Random calibration/test splits of a fixed pool of scores (first N calibration, the rest test).

    Returns, per named index, the thresholds and the number of test scores <= threshold.
    """
    pool = np.asarray(pool, float)
    thr, cnt = {k: [] for k in ms}, {k: [] for k in ms}
    for rng, r in _rngs(key, R, chunk):
        s = pool[np.argsort(rng.random((r, pool.size)), axis=1)]
        t = order_stats(s[:, :N], ms)
        for k in ms:
            thr[k].append(t[k])
            cnt[k].append((s[:, N:] <= t[k][:, None]).sum(axis=1))
    return {"thr": {k: np.concatenate(v) for k, v in thr.items()},
            "count": {k: np.concatenate(v) for k, v in cnt.items()}}


def frac_below(x, c, level=0.99):
    """Fraction of x below c, with an exact (Clopper-Pearson) confidence interval."""
    x = np.asarray(x)
    k, n = int((x < c).sum()), x.size
    ci = st.binomtest(k, n).proportion_ci(confidence_level=level, method="exact")
    return k / n, ci.low, ci.high
