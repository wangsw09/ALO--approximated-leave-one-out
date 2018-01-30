import numpy as np
import numpy.random as npr
import scipy.linalg as spla

def design_iid(size, design_distribution, design_mean=0, design_scale=1,
        design_prob=None, **kwargs):
    """
    Sample the design matrix X iid from a distribution.
    Arguments:
        size: tuple (n, p)
              X will be n by p
        distribution: string, "gauss", or "bern" or "expon"
        args: parameters for the distribution
              "gauss": mean & std
              "bern": p
              "expon": rate
        **kwargs: has no use here, simply serve as a container of the arguments from the
        calling functions.
    Return:
        X: np.ndarray, ndim=2
           Design matrix.
    """
    if design_distribution == "normal":
        return npr.normal(design_mean, design_scale, size=size)
    elif design_distribution == "bern":
        return npr.binomial(1, design_prob, size=size)
    elif design_distribution == "expon":
        return npr.exponential(design_scale, size=size)
    else:
        raise ValueError("Distribution type undefined.")

def design_corr(size, design_scale, design_corr_strength=None,
        design_corr_type="toeplitz", **kwargs):
    """
    Sample the design matrix X from correltaed Gaussian.
    Arguments:
        size: tuple (n, p)
            X will be n by p
        strength: float
            Strength of the correlation
        distribution: string, "gauss", or "bern" or "expon"
        args: parameters for the distribution
            "gauss": mean & std
            "bern": p
            "expon": rate
    Return:
        X: np.ndarray, ndim=2
            Design matrix.
    """
    X = npr.normal(0, design_scale, size=size)
    if design_corr_type == "identity":
        return X
    elif design_corr_type == "toeplitz":
        Sigma_sqrt = spla.sqrtm(spla.toeplitz(design_corr_strength ** np.arange(size[1])))
        return np.dot(X, Sigma_sqrt)
    else:
        raise ValueError("<type> undefined.")

