import numpy as np
import numpy.random as npr

def signal_vec(size, signal_distribution, signal_scale,
        signal_type="dense", signal_sparsity=None, **kwargs):
    if signal_type == "dense":
        if signal_distribution == "normal":
            return npr.normal(0, signal_scale, size=size)
        elif signal_distribution == "expon":
            return npr.exponential(signal_scale, size=size)
    elif signal_type == "sparse":
        beta = np.zeros(size)
        k = int(size * signal_sparsity)
        loc = npr.choice(size, k, replace=False)
        beta[loc] = signal_vec(k, signal_distribution, signal_scale,
                signal_type="dense")
        return beta
    elif signal_type == "positive":
        beta = signal_vec(size, signal_distribution, signal_scale, signal_type="dense")
        return np.fabs(beta)
    else:
        raise ValueError("<type> or <distribution> non-identifiable.")

def signal_mat(size, rank, type="low_rank", **kwargs):
    if type == "low_rank":
        n, p = size
        L = npr.normal(0, 1, size=(n, rank))
        R = npr.normal(0, 1, size=(p, rank))
        d = npr.uniform(0, 1, size = rank)
        return np.dot(L * d, R.T)
    # L.shape=(n, r); d.shape=(r), do broadcasting will work here.

