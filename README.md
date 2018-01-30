# Generalized Approximated Leave-One-Out Cross-Validation

This package implements the **Approximated Leave-One-Out Cross-Validation**(ALO) Algorithms.

## Part II datagen: Data Generation
This module implements the generation of data under various settings.

### Usage
To use the data, call function `datagen.model()` by specifying `size` and `model_spec`:
parameters.
```python
model_spec = {"model_type" : "linear",
              "is_design_iid" : False,
              "design_distribution" : "normal",
              "design_mean" : 0.0,
              "design_scale" : 1.0,
              "design_prob" : None,
              "design_corr_strength" : 0.8,
              "design_corr_type" : "toeplitz",
              "signal_type" : "positive",
              "signal_sparsity" : None,
              "signal_distribution" : "normal",
              "signal_scale" : 3.0,
              "is_noise_iid" : False,
              "noise_tail" : "normal",
              "noise_scale" : 2.0,
              "noise_corr_strength" : 0.9,
              "noise_corr_type" : "toeplitz",
             }
y, X, beta = datagen.model((300, 100), **model_spec)"
```

