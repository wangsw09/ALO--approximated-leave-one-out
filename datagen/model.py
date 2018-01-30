import numpy as np
import numpy.random as npr

from .design import *
from .signal import *
from .noise import *

def response(X, beta, eps=None, model_type="linear"):
    """
    This function generates the response y vector based on design X,
    coeffcient beta, and potential error eps, from the model_type specified.
    """
    if model_type == "linear":
        return np.dot(X, beta) + eps
    elif model_type == "logistic":
        p = 1.0 / (1.0 + np.exp( - np.dot(X, beta)))
        return npr.binomial(1, p)
    else:
        raise ValueError("<model_type> unidentifiable.")

def model(size, model_type="linear", is_design_iid=True, is_noise_iid=True,
        design_distribution=None, signal_type="dense",
        signal_distribution="normal", design_corr_type="toeplitz",
        noise_tail="normal", noise_corr_type="toeplitz", **kwargs):
    """
    kwargs: extra parameters to specify distribution stuff, including:
        design_mean
        design_scale
        design_prob
        design_corr_strength
        design_corr_type
        signal_type
        signal_distribution
        signal_scale
        signal_sparsity
        noise_scale
        noise_corr_strength
    """
    n, p = size
    if is_design_iid is True:
        X = design_iid(size, design_distribution=design_distribution, **kwargs)
    elif is_design_iid is False:
        X = design_corr(size, design_corr_type=design_corr_type, **kwargs)
    beta = signal_vec(p, signal_distribution=signal_distribution,
            signal_type=signal_type, **kwargs)
    
    if model_type == "linear":
        if is_noise_iid is True:
            eps = noise_iid(n, noise_tail=noise_tail, **kwargs)
        else:
            eps = noise_corr(n, noise_corr_type=noise_corr_type, **kwargs)
        y = response(X, beta, eps=eps, model_type="linear")
    elif model_type == "logistic":
        y = response(X, beta, model_type="logistic")

    return y, X, beta

def model_mis():
    return None
