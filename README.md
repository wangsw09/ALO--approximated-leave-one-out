# Approximated Leave-One-Out Cross-Validation (ALO)

This package implements the **Approximated Leave-One-Out Cross-Validation (ALO)** Algorithms.

[Part II datagen: Data Generation](#part-ii-datagen-data-generation)

[Debug: Python 2 vs Python 3 of `glmnet_py`][debug]


## Part II datagen: Data Generation
This module implements the generation of data under various settings.

### Usage
To use the data, call function `datagen.model()` by specifying `size` and `model_spec`:
parameters.
```python
model_spec = {"model_type" : "linear",  # "linear" or "logistic"
              "is_design_iid" : True,  # iid design or not
              "design_distribution" : "normal",  # "normal" or "bern" or "expon"
              "design_mean" : 0.0,  # mean when using Normal design
              "design_scale" : 1.0,  # scale when using Normal or Exponential
              "design_prob" : None,  # prob when using Bernoulli
              "design_corr_strength" : 0.8,  # correlation strength for correlated design
              "design_corr_type" : "toeplitz",  # correlation type, only "toeplitz" now
              "signal_type" : "positive",  # "dense" or "sparse" or "positive"
              "signal_sparsity" : None,  # float between 0 and 1, nonzero loc / p
              "signal_distribution" : "normal",  # "normal" or "expon"
              "signal_scale" : 3.0,  # scale for Normal and Exponential
              "is_noise_iid" : True,  # True or False
              "noise_tail" : "normal",  # tail size, "normal" or "lapace" or "cauchy"
              "noise_scale" : 2.0,  # scale for noise
              "noise_corr_strength" : 0.9,  # noise correlation strength
              "noise_corr_type" : "toeplitz",  # correlation type, only toeplitz now
             }
y, X, beta = datagen.model((300, 100), **model_spec)
```

[debug]## Debug: Python 2 vs Python 3 of `glmnet_py` package
In our package, I call and wrap the functions in package glmnet_py for GLM with elastic-net
type regularizer. Since the package was written in Python 3, there would be some issues
when installed and imported in Python 2 interpreter. In case you are also using our
package in Python 2, below are the changes you need to modify the original glmnet_py
package to make it work.

By the way, Python 3 is the trend, so we may want to switch to it. However I started
this project in Python 2, so I just stick to Python 2 for this one.

* The * parameter in function definition.
In Python 3, you will see such syntax `def bar(*, x, y):`. Here the `*` means that all
the arguments after `*` are required to be used in keyword arguments form.

In Python 2, there are no such syntax and the interpreter will throw a syntax error for
it.

In `glmnet_py`, such syntax appear in the function `glmnet()` in `glmnet_python/glmnet.py`;

Fix: simply remove `*,` from `def glmnet(*, x, y, ...)` to get `def glmnet(x, y, ...)`

* Scope change in list comprehension
In Python 2, when we do list comprehension `[f(x) for x in l1]`, here `x` is visible
from outside, which means after you construct the list and go out of the list
comprehension, `x` will keep the last value in the list `l1`;

In Python 3, this is not true. The scope of `x` is limited to the list comprehension
locally, which becomes unaccessible once the construction finishes.

In `glmnet_python/glmnet.py`, there are several lines use `[x.startwith()... for x in ...]`,
which changes the value of parameter `x` we passed to the function if compiled in
Python 2.

Fix: change the name `x` in these list comprehension by some other names such as
`type_str`.
