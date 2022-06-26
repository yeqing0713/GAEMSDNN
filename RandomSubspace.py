from sklearn.model_selection import KFold
import random

def get_k_random(dim, K, subspace_dim):
    rss = []
    for ii in range(K):
        rs = random.sample(range(0, dim), subspace_dim)
        rss.append(rs)
    return rss

def get_k_random_subspaces(fea, rss):
    fea_subspaces = []
    for ii in range(len(rss)):
        rs = rss[ii]
        fea_subspace = fea[:, rs]
        fea_subspaces.append(fea_subspace)
    return fea_subspaces