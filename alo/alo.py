import numpy as np
import numpy.linalg as npla

from .loss_reg import *
import optim_algo as oa

def alo_glm_ridge(y, X, lam_seq=None, intercept=False,
        family="gaussian", err_func=lambda y1, y2: (y1 - y2) ** 2):
    n, p = X.shape
    m = lam_seq.shape[0]
    alo_err = np.empty(m)

    if intercept is False:
        beta_mat = oa.glm_ridge(y, X, lam_seq=lam_seq, intercept=False,
                family=family)
        U_mat = np.dot(X, beta_mat)
        for i in xrange(m):
            ld, ldd = glm_loss(y, U_mat[:, i], family=family)
            H = np.dot(X, npla.solve(np.dot(X.T * ldd, X) + lam_seq[i] * np.eye(p), X.T))
            y_hat = np.diag(H) / (1.0 - np.diag(H) * ldd) * ld + U_mat[:, i]
            alo_err[i] = np.mean(err_func(y, y_hat))
        return alo_err
    else:
        raise NotImplementedError

def alo_glm_lasso(y, X, lam_seq=None, intercept=False,
        family="gaussian", err_func=lambda y1, y2: (y1 - y2) ** 2):
    n, p = X.shape
    m = lam_seq.shape[0]
    alo_err = np.empty(m)

    if intercept is False:
        beta_mat = oa.glm_lasso(y, X, lam_seq=lam_seq, intercept=False,
                family=family)
        U_mat = np.dot(X, beta_mat)
        for i in xrange(m):
            ld, ldd = glm_loss(y, U_mat[:, i], family=family)
            XE = X[:, beta_mat[:, i] != 0]
            k = XE.shape[1]
            H = np.dot(XE, npla.solve(np.dot(XE.T * ldd, XE), XE.T))
            y_hat = np.diag(H) / (1.0 - np.diag(H) * ldd) * ld + U_mat[:, i]
            alo_err[i] = np.mean(err_func(y, y_hat))
        return alo_err
    else:
        raise NotImplementedError

def alo_L_inf():
    return None

def alo_L_svm():
    return None
