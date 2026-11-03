"""Prescribing a minimum coverage in split conformal prediction.

Split conformal prediction thresholds the non-conformity score of a new point at an order statistic of the
calibration scores: with ``n`` calibration scores sorted as ``s_(1) <= ... <= s_(n)``, the prediction set for an
input ``x`` is ``{y : s(x, y) <= s_(m)}`` for some index ``m``.

- The standard procedure uses ``m* = ceil((n + 1) * nom_cov)``, which guarantees the *marginal* coverage ``nom_cov``.
- The new procedure lets the user prescribe a minimum coverage ``min_cov`` and a confidence ``conf``. It uses the
  smallest ``m`` for which the realised coverage ``C`` of the calibrated predictor satisfies
  ``P(C >= min_cov) >= conf``. For i.i.d. continuous scores ``C ~ Beta(m, n + 1 - m)``, and this index has the closed
  form ``m~ = 1 + Q(conf)``, where ``Q`` is the quantile function of ``Binomial(n, min_cov)``.

Convention: an index ``m = n + 1`` means that no order statistic is large enough. The threshold is then ``+inf``
(a trivial prediction set), and an :class:`InfeasibleRequestWarning` is issued.
"""
import math
import warnings
from fractions import Fraction

import numpy as np
import scipy.stats as st

__all__ = [
    "InfeasibleRequestWarning",
    "classic_conf_pred_wrapper",
    "get_confidence",
    "get_index",
    "get_index_standard",
    "get_mincov",
    "get_minsize",
    "get_threshold",
]


class InfeasibleRequestWarning(UserWarning):
    """The calibration set is too small for the request: the threshold is +inf (trivial prediction set)."""


# ----------------------------------------------------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------------------------------------------------
def _check_n(n):
    if isinstance(n, bool) or not float(n).is_integer() or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    return int(n)


def _check_level(value, name):
    value = float(value)
    if not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value}")
    return value


def _index_new(n, min_cov, conf):
    """Closed form of the new procedure's index, m~ = 1 + Q(conf) with Q the Binomial(n, min_cov) quantile."""
    k = int(st.binom.ppf(conf, n, min_cov))  # smallest k with P(K <= k) >= conf, K ~ Binomial(n, min_cov)
    # guard against round-off in ppf: enforce P(K <= k) >= conf > P(K <= k - 1)
    if st.binom.cdf(k, n, min_cov) < conf:
        k += 1
    elif k > 0 and st.binom.cdf(k - 1, n, min_cov) >= conf:
        k -= 1
    return k + 1  # n + 1 when the request is infeasible, since P(K <= n) = 1


def _index_standard(n, nom_cov):
    """m* = ceil((n + 1) nom_cov) in exact rational arithmetic, capped at n + 1."""
    return min(math.ceil((n + 1) * Fraction(str(nom_cov))), n + 1)


def _msg_new(n, min_cov, conf):
    return (
        f"Infeasible request: with n = {n} calibration points, no order statistic guarantees a minimum coverage of "
        f"{min_cov} with confidence {conf}, so the threshold is +inf (trivial prediction set). The request needs "
        f"at least n = {get_minsize(min_cov, conf)} points. With n = {n}, the largest minimum coverage that can be "
        f"guaranteed with confidence {conf} is {(1 - conf) ** (1 / n):.4f}, and the largest confidence for a minimum "
        f"coverage of {min_cov} is {1 - min_cov ** n:.4f}."
    )


def _msg_standard(n, nom_cov):
    nom = Fraction(str(nom_cov))
    return (
        f"With n = {n} calibration points, the standard procedure with a nominal coverage of {nom_cov} has no "
        f"admissible order statistic, so the threshold is +inf (trivial prediction set). It needs at least "
        f"n = {math.ceil(nom / (1 - nom))} points."
    )


# ----------------------------------------------------------------------------------------------------------------
# Public functions
# ----------------------------------------------------------------------------------------------------------------
def get_index(n, min_cov, conf):
    """Order-statistic index of the new procedure.

    Returns the smallest ``m`` such that the threshold ``s_(m)``, the ``m``-th smallest calibration score,
    guarantees ``P(C >= min_cov) >= conf``. In closed form, ``m = 1 + Q(conf)``, where ``Q`` is the quantile function
    of ``Binomial(n, min_cov)``.

    Parameters
    ----------
    n : int
        Size of the calibration set.
    min_cov : float
        Requested minimum coverage, in (0, 1).
    conf : float
        Confidence ``1 - alpha`` with which the minimum coverage must be achieved, in (0, 1).

    Returns
    -------
    int
        1-based index ``m``. If ``m = n + 1``, no order statistic is large enough: the threshold is ``+inf`` (trivial
        prediction set) and an :class:`InfeasibleRequestWarning` is issued.
    """
    n = _check_n(n)
    min_cov, conf = _check_level(min_cov, "min_cov"), _check_level(conf, "conf")
    m = _index_new(n, min_cov, conf)
    if m > n:
        warnings.warn(_msg_new(n, min_cov, conf), InfeasibleRequestWarning, stacklevel=2)
    return m


def get_index_standard(n, nom_cov):
    """Order-statistic index of the standard procedure, ``m* = ceil((n + 1) * nom_cov)``.

    The product is computed in exact rational arithmetic from the decimal representation of ``nom_cov``, so that
    floating-point round-off cannot select the next order statistic.

    Parameters
    ----------
    n : int
        Size of the calibration set.
    nom_cov : float
        Nominal (marginal) coverage, in (0, 1).

    Returns
    -------
    int
        1-based index ``m*``. If ``m* > n``, the function returns ``n + 1`` (threshold ``+inf``, trivial prediction set)
        and issues an :class:`InfeasibleRequestWarning`.
    """
    n = _check_n(n)
    nom_cov = _check_level(nom_cov, "nom_cov")
    m = _index_standard(n, nom_cov)
    if m > n:
        warnings.warn(_msg_standard(n, nom_cov), InfeasibleRequestWarning, stacklevel=2)
    return m


def get_threshold(scores, min_cov, conf):
    """Threshold of the new procedure: the ``m~``-th smallest calibration score.

    The prediction set for a new input ``x`` is ``{y : s(x, y) <= threshold}``.

    Parameters
    ----------
    scores : array_like
        Non-conformity scores of the calibration set (flattened if multidimensional).
    min_cov : float
        Requested minimum coverage, in (0, 1).
    conf : float
        Confidence ``1 - alpha``, in (0, 1).

    Returns
    -------
    float
        The threshold ``s_(m~)``, or ``+inf`` (with an :class:`InfeasibleRequestWarning`) if the calibration set is too
        small for the request.
    """
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0 or np.isnan(s).any():
        raise ValueError("scores must be a non-empty array without NaN")
    min_cov, conf = _check_level(min_cov, "min_cov"), _check_level(conf, "conf")
    m = _index_new(s.size, min_cov, conf)
    if m > s.size:
        warnings.warn(_msg_new(s.size, min_cov, conf), InfeasibleRequestWarning, stacklevel=2)
        return np.inf
    return float(np.partition(s, m - 1)[m - 1])


def get_confidence(n, m, min_cov):
    """Confidence achieved by the threshold ``s_(m)``: ``P(C >= min_cov)`` with ``C ~ Beta(m, n + 1 - m)``.

    It equals ``P(K <= m - 1)`` for ``K ~ Binomial(n, min_cov)``. For ``m = n + 1`` (trivial prediction set) it is 1.

    Parameters
    ----------
    n : int
        Size of the calibration set.
    m : int
        1-based order-statistic index, between 1 and ``n + 1``.
    min_cov : float
        Minimum coverage, in (0, 1).

    Returns
    -------
    float
    """
    n = _check_n(n)
    min_cov = _check_level(min_cov, "min_cov")
    if isinstance(m, bool) or not float(m).is_integer() or not 1 <= m <= n + 1:
        raise ValueError(f"m must be an integer between 1 and n + 1 = {n + 1}, got {m!r}")
    return 1.0 if m == n + 1 else float(st.binom.cdf(int(m) - 1, n, min_cov))


def get_mincov(n, conf, nom_cov=None, min_cov=None):
    """Minimum coverage guaranteed with confidence ``conf``: the ``(1 - conf)``-quantile of the realised coverage.

    Give exactly one of

    - ``nom_cov``: the standard procedure with nominal coverage ``nom_cov`` (index ``m*``);
    - ``min_cov``: the new procedure with requested minimum coverage ``min_cov`` (index ``m~``). The result is the
      achieved bound ``C^min* >= min_cov``, which exceeds ``min_cov`` because the index is an integer.

    Parameters
    ----------
    n : int
        Size of the calibration set.
    conf : float
        Confidence, in (0, 1).
    nom_cov, min_cov : float, optional
        Nominal coverage (standard procedure) or requested minimum coverage (new procedure).

    Returns
    -------
    float
        The guaranteed minimum coverage; 1.0 when the prediction set is trivial (no admissible order statistic, with
        an :class:`InfeasibleRequestWarning`).
    """
    if (nom_cov is None) == (min_cov is None):
        raise ValueError("give exactly one of nom_cov (standard procedure) or min_cov (new procedure)")
    n = _check_n(n)
    conf = _check_level(conf, "conf")
    if nom_cov is not None:
        nom_cov = _check_level(nom_cov, "nom_cov")
        m = _index_standard(n, nom_cov)
        message = _msg_standard(n, nom_cov) if m > n else None
    else:
        min_cov = _check_level(min_cov, "min_cov")
        m = _index_new(n, min_cov, conf)
        message = _msg_new(n, min_cov, conf) if m > n else None
    if message is not None:
        warnings.warn(message, InfeasibleRequestWarning, stacklevel=2)
        return 1.0
    return float(st.beta.ppf(1 - conf, m, n + 1 - m))


def get_minsize(min_cov, conf, nom_cov=None):
    """Minimum calibration-set size that guarantees ``P(C >= min_cov) >= conf``.

    - ``nom_cov=None`` (default): the new procedure. The request is feasible exactly when
      ``1 - min_cov**n >= conf`` (Wilks' condition), so the minimum size is ``ceil(ln(1 - conf) / ln(min_cov))``.
    - ``nom_cov`` given: the standard procedure with nominal coverage ``nom_cov``. Its confidence ``P(C >= min_cov)``
      is a sawtooth function of ``n``, so the function returns the smallest ``n0`` such that every ``n >= n0`` meets
      the requirement (checked numerically up to at least twice ``n0``, and far beyond the normal-approximation
      estimate). Requires ``nom_cov > min_cov``.

    Parameters
    ----------
    min_cov : float
        Minimum coverage, in (0, 1).
    conf : float
        Confidence, in (0, 1).
    nom_cov : float, optional
        Nominal coverage of the standard procedure, in (0, 1).

    Returns
    -------
    int
    """
    min_cov, conf = _check_level(min_cov, "min_cov"), _check_level(conf, "conf")
    if nom_cov is None:
        n = max(1, math.ceil(math.log1p(-conf) / math.log(min_cov)))
        while 1 - min_cov ** n < conf:  # guard against round-off in the logarithms
            n += 1
        while n > 1 and 1 - min_cov ** (n - 1) >= conf:
            n -= 1
        return n

    nom = Fraction(str(_check_level(nom_cov, "nom_cov")))
    if nom <= Fraction(str(min_cov)):
        raise ValueError("the standard procedure guarantees P(C >= min_cov) >= conf for all large n only if "
                         "nom_cov > min_cov")
    a, b = nom.numerator, nom.denominator
    z = max(st.norm.ppf(conf), 1.0)
    n_est = (z * math.sqrt(min_cov * (1 - min_cov)) / (float(nom) - min_cov)) ** 2  # normal approximation
    n_max = int(max(1_000, 20 * n_est))
    while True:
        n = np.arange(1, n_max + 1, dtype=np.int64)
        m = -((-(n + 1) * a) // b)  # ceil((n + 1) * nom_cov), exact integer arithmetic
        ok = (m <= n) & (st.binom.cdf(m - 1, n, min_cov) >= conf)
        failing = np.flatnonzero(~ok)
        last_failing = int(failing[-1]) + 1 if failing.size else 0
        if last_failing <= n_max // 2:
            return last_failing + 1
        if n_max >= 5_000_000:
            raise RuntimeError("no minimum size found below 5e6; nom_cov is too close to min_cov")
        n_max *= 4


def classic_conf_pred_wrapper(n, min_cov, conf):
    """Confidence level that makes a standard split-CP implementation use the new procedure's threshold.

    Standard split CP selects the order statistic ``ceil((n + 1) * level)``. The returned level,
    ``(m~ - 0.5) / (n + 1)``, makes that rule select exactly ``m~``, with a margin of half an index against
    floating-point round-off. Pass it, for example, as the ``confidence_level`` of a MAPIE split conformal regressor.

    Parameters
    ----------
    n : int
        Size of the calibration set.
    min_cov : float
        Requested minimum coverage, in (0, 1).
    conf : float
        Confidence ``1 - alpha``, in (0, 1).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the request is infeasible: no finite threshold meets it, so there is no level to pass on.
    """
    n = _check_n(n)
    min_cov, conf = _check_level(min_cov, "min_cov"), _check_level(conf, "conf")
    m = _index_new(n, min_cov, conf)
    if m > n:
        raise ValueError(_msg_new(n, min_cov, conf))
    return (m - 0.5) / (n + 1)
