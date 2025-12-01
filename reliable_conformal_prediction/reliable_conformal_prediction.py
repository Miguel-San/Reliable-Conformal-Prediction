import numpy as np

import scipy.stats as st
from scipy.optimize import root_scalar

def get_quantile(n:int, min_cov:float, conf:float, tol:float=1e-3, max_iter:int=1000):
    """
    Function that searches for the quantile level that give a minimum coverage with a given confidence and with a given sample size.
    Sample quantiles are supposed to be computed with order statistics

    Parameters
    ----------
    n : int
        Size of the calibration set
    min_cov : float
        Minimum coverage level to be achieved
    conf : float
        Confidence level
    tol : float, optional
        Tolerance (absolute) for the root finding algorithm, by default 1e-3
    max_iter : int, optional
        Maximum number of iterations for the root finding algorithm, by default 1000
    """
    def is_mincov_achieved(delta:float, min_cov:float):
        i = delta + 1  # +1 to avoid zero index
        j = n - delta

        alpha = j
        beta = n - j + 1

        coverage_dist = st.beta(alpha, beta)
        return (coverage_dist.cdf(min_cov)) - (1 - conf)

    try:
        opt_result = root_scalar(
            lambda x: is_mincov_achieved(x, min_cov),
            bracket=[0, np.ceil(n*(1 - min_cov))],
            maxiter=max_iter,
            xtol=tol
        )
    except:
        return np.nan

    p_level = 1 - (np.floor(opt_result.root) / n)
    return p_level

def get_minsize(min_cov:float, nom_cov:float, conf:float, approx="upper"):
    """
    Function that searches for the minimum calibration size to achieve a desired nominal and minimum coverage with a given confidence
    Sample quantiles are assumed to be computed using order statistics.
    The `approx` parameter specifies the approximation method to use. 
    """

    def is_mincov_achieved(n:int):
        if approx == "lower":
            i = n * nom_cov + 1
        elif approx == "upper":
            i = n * nom_cov
        else:
            raise ValueError("Approximation method not recognized. Use 'lower' or 'upper'.")

        alpha = i
        beta = n - i + 1

        coverage_dist = st.beta(alpha, beta)
        return (1 - conf) - (coverage_dist.cdf(min_cov))

    opt_result = root_scalar(
        lambda x: is_mincov_achieved(x),
        x0=100
    )

    return np.ceil(opt_result.root)

def get_mincov(n:int, nom_cov:float, conf:float):
    """
    Function that gives the minimum coverage witha given confidence for a given sample size and nominal coverage.
    Sample quantiles are assumed to be computed using order statistics.
    """

    alpha = np.floor(n*nom_cov)
    beta = n - np.floor(n*nom_cov) + 1

    cov_dist = st.beta(a=alpha, b=beta)
    return cov_dist.ppf(1-conf)

def classic_conf_pred_wrapper(n:int, min_cov:float, conf:float):
    """
    Wrapper function that returns the quantile levels for classic conformal prediction that achieve a minimum coverage with a given confidence and sample size.

    Parameters
    ----------
    n : int
        Size of the calibration set
    min_cov : float
        Minimum coverage level to be achieved
    conf : float
        Confidence level
    """
    p_level = get_quantile(n, min_cov, conf)
    return p_level*n/(n+1)  # Adjustment for classic conformal prediction
