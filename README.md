# Reliable Conformal Prediction

MIT License

Copyright (c) 2025 Miguel Sánchez-Domínguez, Lucas Lacasa, Javier de Vicente, Gonzalo Rubio, Eusebio Valero

### [Access to Paper](https://arxiv.org/pdf/2512.04566)

Surrogate models --including deep neural networks and other machine learning algorithms in supervised learning-- are capable of approximating arbitrarily complex, high-dimensional input-output problems in science and engineering, but require a thorough data-agnostic uncertainty quantification analysis before these can be deployed for any safety-critical application. The standard approach for data-agnostic uncertainty quantification is to use conformal prediction (CP), a well-established framework to build uncertainty models with proven statistical guarantees that do not assume any shape for the error distribution of the surrogate model. However, since the classic statistical guarantee offered by CP is given in terms of bounds for the marginal (i.e. ensemble-averaged) coverage, for small calibration set sizes --which are frequent in realistic surrogate modelling that aims to quantify error at different regions--, the potentially strong dispersion of the coverage distribution around its average negatively impacts the reliability of the uncertainty model, often obtaining coverages below the expected value, resulting in a less applicable framework. 

After providing a gentle presentation of uncertainty quantification for surrogate models for machine learning practitioners, in [this paper](https://arxiv.org/pdf/2512.04566) we bridge the gap by proposing a new statistical guarantee that offers probabilistic information for the coverage of a single conformal predictor. 
We show that the proposed framework converges to the standard solution offered by CP for large calibration set sizes and, unlike the classic guarantee, still offers reliable information about the coverage of a conformal predictor for small data sizes. 
This repository is the open access, user-friendly software implementation of the methods used buld conformal predictors that achieve the new guarantee. These methods can be used alongside common conformal prediction libraries like MAPIE.

## Installation

The methods implemented in this repo can be easily installed as a Python package using

```console
pip install git+https://github.com/Miguel-San/Reliable-Conformal-Prediction.git
```

Once installed, methods can be accessed by
```python
import reliable_conformal_prediction
```

In the folder `Examples` of this repository you can find some simple usage examples for the methods implemented in this package.

## Citation
If any part of our paper and/or repository is helpful to your work, we would appreciate citations with:
```
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