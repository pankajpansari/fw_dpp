import networkx as nx
import numpy as np
import sys
import time
import random
import torch
import os
import math
import itertools as it
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
        return L

    def cache_reset(self):
        self.cache.reset()
        self.cache_hits = 0

    def counter_reset(self):
        self.itr_total = 0
        self.itr_new = 0
        self.itr_cache = 0

    def __call__(self, sample):

        key = sample.tobytes()
        self.itr_total += 1 
        if key not in self.cache:
            self.itr_new += 1 
            val = getDet(self.L, sample)
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
    temp = subMat.data.numpy()
    detVal = np.linalg.det(temp)
    temp = np.array([detVal])
    return torch.from_numpy(temp)

 
def main():
    N = 100 
    L = read_dpp("/home/pankaj/Sampling/data/input/dpp/data/dpp_100_1.5_0.5_200_0_0.1_5.h5", N, '/dpp_0')
    sample = torch.rand(N) > 0.8

    dpp_obj = DPP(L)

    print dpp_obj(sample.numpy())

if __name__ == '__main__':
    main()
