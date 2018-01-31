import numpy as np
import numpy.random as npr
import glmnet_python
from glmnet import glmnet

def glm_ridge(y, X, lam=None, lam_seq=None, intercept=False, family="gaussian"):
    """
    We are going to simply wrap the functions in glmnet.
    """
    if lam is not None:
        if not intercept:
            beta = glmnet(x=X, y=y, family=family, intr=False, alpha=0, nlambda=1,
                    lambdau=np.array([lam]))['beta'][:, 0]
            return beta
        else:
            res = glmnet(x=X, y=y, family=family, intr=True, alpha=0, nlambda=1,
                    lambdau=np.array([lam]))
            return res['beta'][:, 0], res['a0'][0]
    elif lam_seq is not None:
        if not intercept:
            beta = glmnet(x=X, y=y, family=family, intr=False, alpha=0, nlambda=1,
                    lambdau=np.asarray(lam_seq))['beta']
            return beta
        else:
            res = glmnet(x=X, y=y, family=family, intr=True, alpha=0, nlambda=1,
                    lambdau=np.asarray(lam_seq))
            return res['beta'], res['a0']
 


