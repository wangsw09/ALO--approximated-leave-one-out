import numpy as np
import numpy.random as npr
import scipy.linalg as spla

def noise_iid(size, noise_scale, noise_tail="normal", **kwargs):
    if noise_tail == "normal":
        return npr.normal(0, 1, size=size) * noise_scale
    elif noise_tail == "laplace":
        return npr.laplace(0, 1, size=size) * noise_scale
    elif noise_tail == "cauchy":
        return npr.standard_cauchy(size=size) * noise_scale

def noise_corr(size, noise_scale, noise_corr_strength,
        noise_corr_type="toeplitz", **kwargs):
    eps = npr.normal(0, noise_scale, size=size)
    if noise_corr_type == "toeplitz":
        Sigma = spla.sqrtm(spla.toeplitz(noise_corr_strength ** np.arange(size)))
        return np.dot(Sigma, eps)
    if noise_corr_type == "ar":
        raise ValueError("To be implemented.")
