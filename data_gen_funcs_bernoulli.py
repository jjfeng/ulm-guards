import numpy as np

def curvy_bin_mu(xs):
    raw_mu = np.sin(xs[:,0] * 0.5) + np.sin(0.2*xs[:,1]) + np.sin(0.1 * xs[:,2] + 0.05*xs[:,3]) + np.sin(0.1 * xs[:,3])
    return 0.49 * np.tanh(raw_mu) + 0.5
