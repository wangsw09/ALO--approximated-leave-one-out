import numpy as np
import numpy.linalg as npla

from .loss_reg import *
import optim_algo as oa

def loo_glm_ridge(y, X, lam_seq=None, intercept=False,
        family="gaussian", err_func=lambda y1, y2: (y1 - y2) ** 2 ):
    n, p = X.shape
    m = lam_seq.shape[0]
    loo_err = np.zeros(m)

    mask = np.ones(n, dtype=np.bool)

    if intercept is False:
        for i in xrange(n):
            mask[i] = False
            beta_mat = oa.glm_ridge(y[mask], X[mask, :], lam_seq=lam_seq, intercept=False,
                family=family)
            pred = np.dot(X[i, :], beta_mat)
            loo_err += err_func(y[i], pred)
            mask[i] = True
        return loo_err / n

    else:
        raise NotImplementedError

def loo_glm_lasso(y, X, lam_seq=None, intercept=False,
        family="gaussian", err_func=lambda y1, y2: (y1 - y2) ** 2 ):
    n, p = X.shape
    m = lam_seq.shape[0]
    loo_err = np.zeros(m)

    mask = np.ones(n, dtype=np.bool)

    if intercept is False:
        for i in xrange(n):
            mask[i] = False
            beta_mat = oa.glm_lasso(y[mask], X[mask, :], lam_seq=lam_seq, intercept=False,
                family=family)
            pred = np.dot(X[i, :], beta_mat)
            loo_err += err_func(y[i], pred)
            mask[i] = True
        return loo_err / n

    else:
        raise NotImplementedError


