import numpy as np
import numpy.random as npr
from scipy.linalg import eigh as eig_max
import glmnet_python
from glmnet import glmnet

from .proximal import *

def glm_lasso(y, X, lam=None, lam_seq=None, intercept=False, family="gaussian"):
    """
    We are going to simply wrap the functions in glmnet.
    """
    if lam is not None:
        if not intercept:
            beta = glmnet(x=X, y=y, family=family, intr=False, alpha=1, nlambda=1,
                    lambdau=np.array([lam]))['beta'][:, 0]
            return beta
        else:
            res = glmnet(x=X, y=y, family=family, intr=True, alpha=1, nlambda=1,
                    lambdau=np.array([lam]))
            return res['beta'][:, 0], res['a0'][0]
    elif lam_seq is not None:
        if not intercept:
            beta = glmnet(x=X, y=y, family=family, intr=False, alpha=1, nlambda=1,
                    lambdau=np.asarray(lam_seq))['beta']
            return beta
        else:
            res = glmnet(x=X, y=y, family=family, intr=True, alpha=1, nlambda=1,
                    lambdau=np.asarray(lam_seq))
            return res['beta'], res['a0']
 
def slope(y, X, lam, weights, intercept=False):
    return None

def L_inf(y, X, lam, intercept=False, tol=1e-5, max_iter=500):
    n, p = X.shape
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    ss = 0.9 / eig_max(XTX, eigvals=(p-1, p-1), eigvals_only=True)[0]
    beta0 = np.ones(p)
    beta = np.zeros(p)
    
    iter_count = 0

    while np.amax(np.fabs(beta - beta0)) > tol and iter_count < max_iter:
        np.copyto(beta0, beta)
        beta = L_inf_prox(beta - ss * np.dot(XTX, beta) + ss * XTy, ss * lam)
        iter_count += 1

    print iter_count
    return beta


