# Approximated Leave-One-Out Cross-Validation (ALO)

This package implements the **Approximated Leave-One-Out Cross-Validation (ALO)** Algorithms.

## Part II datagen: Data Generation
This module implements the generation of data under various settings.

### Usage
To use the data, call function `datagen.model()` by specifying `size` and `model_spec`:
parameters.
```python
model_spec = {"model_type" : "linear",  # "linear" or "logistic"
              "is_design_iid" : False,  # True or False
              "design_distribution" : "normal",  # "normal" or "bern" or "expon"
              "design_mean" : 0.0,  # mean when using Normal
              "design_scale" : 1.0,  # scale when using Normal or Exponential
              "design_prob" : None,  # prob when using Bernoulli
              "design_corr_strength" : 0.8,  # correlation strength for correlated design
              "design_corr_type" : "toeplitz",  # correlation type, only "toeplitz" now
              "signal_type" : "positive",  # "dense" or "sparse" or "positive"
              "signal_sparsity" : None,  # float between 0 and 1, nonzero loc / p
              "signal_distribution" : "normal",  # "normal" or "expon"
              "signal_scale" : 3.0,  # scale for Normal and Exponential
              "is_noise_iid" : False,  # True or False
              "noise_tail" : "normal",  # tail size, "normal" or "lapace" or "cauchy"
              "noise_scale" : 2.0,  # scale for noise
              "noise_corr_strength" : 0.9,  # noise correlation strength
              "noise_corr_type" : "toeplitz",  # correlation type, only toeplitz now
             }
y, X, beta = datagen.model((300, 100), **model_spec)"
```

