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

def alo_L_inf(y, X, lam_seq=None, intercept=False, err_func=lambda y1, y2: (y1 - y2) ** 2):
    n, p = X.shape
    m = lam_seq.shape[0]
    alo_err = np.empty(m)

    if intercept is False:
        for i in xrange(m):
            beta = oa.L_inf(y, X, lam=lam_seq[i], intercept=False,
                    max_iter=1000)
            u = np.dot(X, beta)
            ld, ldd = glm_loss(y, u, family="gaussian")
            XTmu = np.dot(X.T, y - u)

            A = (np.fabs(XTmu) > 1e-6)
            W = np.empty((n, p - np.sum(A) + 1))
            W[:, :(-1)] = X[:, np.logical_not(A)]
            W[:, -1] = np.sum(X[:, A] * np.sign(XTmu[A]), axis=1)

            H = np.dot(W, npla.solve(np.dot(W.T * ldd, W), W.T))
            y_hat = np.diag(H) / (1.0 - np.diag(H) * ldd) * ld + u
            alo_err[i] = np.mean(err_func(y, y_hat))
        return alo_err
    else:
        raise NotImplementedError

def alo_svm(y, X, lam_seq=None, intercept=False, tol_tr=1e-6, tol_sg=1e-5,
        max_iter=2000, err_func=lambda y1, y2: (y1 - y2) ** 2):
    n, p = X.shape
    m = lam_seq.shape[0]
    alo_err = np.empty(m)

    if intercept is False:
        for i in xrange(m):
            beta = oa.svm_linear(y, X, lam=lam_seq[i], tol=tol_tr,
                    max_iter=max_iter, intercept=False)
            u = np.dot(X, beta) * y
            V = (np.fabs(u - 1.0) < tol_sg)
            S = np.logical_not(V)
            
            a = np.empty(n)
            g = np.empty(n)
            
            if np.any(V):
                Y = npla.inv(np.dot(X[V, :], X[V, :].T))
                
                a[S] = np.diag(np.dot(np.dot(X[S, :], np.eye(p) -
                    np.dot(np.dot(X[V, :].T, Y), X[V, :])), X[S, :].T)) / lam_seq[i]
                a[V] = 1.0 / lam_seq[i] / np.diag(Y)
                
                g[S] = np.where(u[S] > 1, 0, -y[S])
                g[V] = npla.lstsq(X[V, :].T, - np.dot(X[S, :].T, g[S]) -
                        lam_seq[i] * beta)[0]
            
            else:
                a = np.diag(np.dot(X, X.T)) / lam_seq[i]
                g = np.where(u > 1, 0, -y[S])
                
            y_hat = np.dot(X, beta) + a * g
            alo_err[i] = np.mean(err_func(y, y_hat))
        return alo_err
    else:
        raise NotImplementedError


