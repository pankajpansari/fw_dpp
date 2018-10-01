import networkx as nx
import numpy as np
import sys
import time
import random
import torch
import os
import math
import itertools
import matplotlib.pyplot as plt
from torch.autograd import Variable
from read_files import read_dpp

random.seed(1234)

class DPP(object):
    def __init__(self, qualities, features):
        self.cache = {}
        self.cache_hits = 0
        self.qualities = qualities #vector of size #items (N)
        self.features =  features #matrix of size #features(D)XN
        self.N = qualities.shape[0]
        self.L = self.getMatrix(qualities, features)
        self.itr_total = 0
        self.itr_new = 0
        self.itr_cache = 0

    def getMatrix(self, qualities, features):
        D = features.size()[0]
        B = torch.zeros((D, self.N))
        for t in range(self.N):
            B[:, t] = qualities[t]*features[:, t]
        L = torch.mm(B.t(), B)

        w, v = np.linalg.eig(L.numpy())
        assert (w >= -1e-10).all(), "Negative eigenvalue" #comparison with -1e-10 because of numerical issues in computing the eigenvalues
        return L

    def enumerate(self):
        power_set = map(list, itertools.product([0, 1], repeat= self.N))

        for binary_vec in power_set:

            sample = torch.from_numpy(np.array(binary_vec)).float()

            f_val = torch.abs(self(sample))

            this_set = torch.LongTensor([i for i in range(self.N) if sample[i] == 1] )
            print this_set.numpy(), ": ", f_val.item()

    def cache_reset(self):
        self.cache.reset()
        self.cache_hits = 0

    def counter_reset(self):
        self.itr_total = 0
        self.itr_new = 0
        self.itr_cache = 0

    def __call__(self, sample):

        key = sample.numpy().tobytes()
        self.itr_total += 1 
        eps = 1e-4
        if key not in self.cache:
            self.itr_new += 1 
            val = torch.log(getDet(self.L, sample) + eps)
            if torch.isnan(val):
                print "submodular function is nan"
                print val, getDet(self.L, sample)
                sys.exit()
            self.cache[key] = val
        else:
            self.itr_cache += 1 
            self.cache_hits += 1

        return self.cache[key]

###########################################
def getDet(L, sample):

    N = L.shape[0]
    this_set = torch.LongTensor([i for i in range(N) if sample[i] == 1] )
    if this_set.nelement() == 0:
        return torch.Tensor([1])

    subRowsMatrix = L.index_select(0, this_set)
    subMat = subRowsMatrix.index_select(1, this_set)
    return subMat.det()
 
if __name__ == '__main__':
    (qualities, features) = read_dpp('/home/pankaj/Sampling/data/input/dpp/data/dpp_10_10_1_5_3_0_0.1.h5', 'dpp_0')
    dpp = DPP(qualities, features)
    sample = torch.Tensor([0]*dpp.N)
    sample[0] = 1
    sample[2] = 1
    this_set = torch.LongTensor([i for i in range(dpp.N) if sample[i] == 1])
    subRowsMatrix = dpp.L.index_select(0, this_set)
    subMat = subRowsMatrix.index_select(1, this_set)
    print subMat
    print dpp(sample)
