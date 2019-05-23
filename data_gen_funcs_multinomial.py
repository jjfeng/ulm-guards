import numpy as np

def category3_mu(xs):
    raw_mu1 = 0.5 * xs[:,0]
    raw_mu2 = 0.5 * xs[:,1]
    raw_mu3 = 0.5 * xs[:,2]
    new_mu = np.exp(np.vstack([raw_mu1, raw_mu2, raw_mu3]))
    sum_mu = np.sum(new_mu, axis=0).reshape((-1,1))
    return np.transpose(new_mu)/sum_mu
