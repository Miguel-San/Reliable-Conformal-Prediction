# Reliable Conformal Prediction

MIT License

Copyright (c) 2025 Miguel Sánchez-Domínguez, Lucas Lacasa, Javier de Vicente, Gonzalo Rubio, Eusebio Valero

### [Access to Paper](https://arxiv.org/pdf/2512.04566)

Surrogate models --including deep neural networks and other machine learning algorithms in supervised learning-- are capable of approximating arbitrarily complex, high-dimensional input-output problems in science and engineering, but require a thorough data-agnostic uncertainty quantification analysis before these can be deployed for any safety-critical application. The standard approach for data-agnostic uncertainty quantification is to use conformal prediction (CP), a well-established framework to build uncertainty models with proven statistical guarantees that do not assume any shape for the error distribution of the surrogate model. However, since the classic statistical guarantee offered by CP is given in terms of bounds for the marginal (i.e. ensemble-averaged) coverage, for small calibration set sizes --which are frequent in realistic surrogate modelling whose prediction error needs to be quantified at different regions--, the potentially strong dispersion of the coverage distribution around its average negatively impacts the relevance of the uncertainty model's statistical guarantee, often obtaining coverages below the expected value, resulting in a less applicable framework.

After providing a gentle presentation of uncertainty quantification for surrogate models for machine learning practitioners, in [this paper](https://arxiv.org/pdf/2512.04566) we bridge the gap by proposing a new statistical guarantee that offers probabilistic information for the coverage of a single conformal predictor. 
We show that the proposed framework converges to the standard solution offered by CP for large calibration set sizes and, unlike the classic guarantee, still offers relevant information about the coverage of a conformal predictor for small data sizes. 
This repository is the open access, user-friendly software implementation of the methods used to build conformal predictors that achieve the new guarantee. These methods can be used alongside common conformal prediction libraries like MAPIE.

## Installation

The methods implemented in this repo can be easily installed as a Python package using

```console
pip install git+https://github.com/Miguel-San/Reliable-Conformal-Prediction.git@v2.0.0
```

Once installed, methods can be accessed by
```python
import reliable_conformal_prediction
```

In the folder `Examples` of this repository you can find some simple usage examples for the methods implemented in this package.

## Usage

The new procedure takes a requested minimum coverage `min_cov` and a confidence `conf`, and uses the smallest order statistic of the calibration scores that guarantees `P(C >= min_cov) >= conf` for the realised coverage `C`. This index has the closed form `1 + Q(conf)`, where `Q` is the quantile function of `Binomial(n, min_cov)`.

```python
import numpy as np
import reliable_conformal_prediction as rcp

scores = np.abs(y_cal - model.predict(X_cal))  # non-conformity scores of the calibration set
n, min_cov, conf = len(scores), 0.9, 0.95

m = rcp.get_index(n, min_cov, conf)                   # order statistic of the new procedure
threshold = rcp.get_threshold(scores, min_cov, conf)  # the m-th smallest score
# prediction interval for a new input x: model.predict(x) -/+ threshold

rcp.get_minsize(min_cov, conf)              # minimum calibration size for this request (29)
rcp.get_mincov(n, conf, min_cov=min_cov)    # achieved minimum coverage, C^min* >= min_cov
rcp.get_confidence(n, m, min_cov)           # achieved confidence, P(C >= min_cov) >= conf
rcp.get_mincov(n, conf, nom_cov=0.9)        # what the standard procedure guarantees at the same confidence
```

If the calibration set is too small for the request (`n < get_minsize(min_cov, conf)`), no order statistic is large enough: `get_index` returns `n + 1`, `get_threshold` returns `inf` (a trivial prediction set), and both issue an `InfeasibleRequestWarning` that reports the minimum size and what the data can certify.

With [MAPIE](https://github.com/scikit-learn-contrib/MAPIE), `classic_conf_pred_wrapper` gives the confidence level that makes a split conformal regressor use the new procedure's threshold:

```python
from mapie.regression import SplitConformalRegressor

level = rcp.classic_conf_pred_wrapper(n, min_cov, conf)
mapie = SplitConformalRegressor(estimator=model, confidence_level=level, prefit=True)
mapie.conformalize(X_cal, y_cal)
```

## Changelog

**2.0.0**
- The order statistic of the new procedure is computed with the exact closed form `1 + Q(conf)` instead of root finding. The root finding disagreed with the exact index in a few cases where the root lies very close to an integer (one of them below the requested confidence), and returned the level 0 for `n = 1`.
- `get_quantile` is replaced by `get_index` (the order-statistic index) and `get_threshold` (the threshold itself, from the calibration scores), so that quantile levels are never converted back into thresholds.
- Infeasible requests return the index `n + 1` (threshold `inf`) with an `InfeasibleRequestWarning`, instead of `NaN` without a warning. `classic_conf_pred_wrapper` raises a `ValueError` for them.
- `get_minsize(min_cov, conf)` gives the minimum calibration size of the new procedure, and `get_minsize(min_cov, conf, nom_cov=...)` that of the standard procedure. `get_mincov` takes `nom_cov` (standard procedure) or `min_cov` (new procedure).
- New functions `get_index_standard` and `get_confidence`.

**1.0.0**
- First release.

## Citation
If any part of our paper and/or repository is helpful to your work, we would appreciate citations with:
```bibtex
@article{sanchezdominguez2025reliable,
  title={Reliable Statistical Guarantees for Conformal Predictors with Small Datasets},
  author={S{\'a}nchez-Dom{\'\i}nguez, Miguel and Lacasa, Lucas and de Vicente,
          Javier and Rubio, Gonzalo and Valero, Eusebio},
  year={2025},
  eprint={2512.04566},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2512.04566},
}
```
