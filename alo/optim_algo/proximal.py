import numpy as np

def L_inf_prox(z, tau):
    p = z.shape[0]
    zs = np.fabs(z)
    z_rank = np.argsort(zs)
    zs = np.sort(zs)

    xmax = zs[-1] - tau
    i = 1
    while xmax <= zs[-(i + 1)]:
        xmax = (xmax * i + zs[-(i + 1)]) / (i + 1.0)
        i += 1
        if i + 1 > p:
            break
    zs[(-i) : ] = xmax
    tmp = np.empty(p)
    tmp[z_rank] = zs
    return np.copysign(tmp, z)
