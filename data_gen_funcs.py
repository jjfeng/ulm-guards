import numpy as np

def curvy_2d_mu(xs):
    return np.sin(0.05 * np.power(xs[:,0], 2) + 0.05 * np.power(xs[:,1], 2))

def curvy_2d_sigma(xs):
    return 1.5 * np.abs(np.sin(0.05 * np.power(xs[:,0], 2))) + 0.1

def simple_2d_mu(xs):
    return np.abs(0.5 * xs[:,0] + xs[:,1]) - 0.5 * (np.abs(xs[:,1]) + np.abs(xs[:,0]))

def simple_2d_sigma(xs):
    return 0.08 * (np.abs(xs[:,0] + 3) + np.abs(xs[:,1] - 3)) + 0.1

def constant_2d_mu(xs):
    return np.abs(0.5 * xs[:,0] + xs[:,1]) - 0.5 * (np.abs(xs[:,1]) + np.abs(xs[:,0]))

def constant_2d_sigma(xs):
    return np.ones(xs.shape[0])

def linear_mu(xs):
    raw_mu = xs[:,0] + 2*xs[:,1] + xs[:,2] + 2*xs[:,3]
    return raw_mu

def linear_sigma(xs):
    sigma_seq = 0.3 * np.abs(xs[:,0] + 2*xs[:,1] + xs[:,2] + 2*xs[:,3]) + 0.1
    return sigma_seq

def simple_mu(xs):
    return np.tanh(xs[:,0] + 2*xs[:,1]) + xs[:,0] + np.tanh(xs[:,2] + 2*xs[:,3])

def simple_sigma(xs):
    return np.tanh(xs[:,0] + 2*xs[:,1]) + 1.1

def curvy_mu(xs):
    return np.sin(xs[:,0] + 2*xs[:,1]) + xs[:,0] + 2 * np.sin(xs[:,2] + 2*xs[:,3])

def curvy_sigma(xs):
    return np.sin(xs[:,0] + 2*xs[:,1]) + 1.1

def simple_surprise_2d_mu(xs):
    inside_surprise = (np.abs(xs[:,0]) < 1) * (np.abs(xs[:,1]) < 1)
    outside_surprise = (1 - inside_surprise)
    return (0.5 * xs[:,0] + xs[:,1]) * outside_surprise + np.power(xs[:,0] * xs[:,1], 2) * inside_surprise

def simple_surprise_2d_sigma(xs):
    return 0.4 * np.ones(xs.shape[0])
