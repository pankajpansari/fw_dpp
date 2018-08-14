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

random.seed(1234)

dirw = "/home/pankaj/Sampling/data/working/17_07_2018/"

class Influence(object):
    def __init__(self, G, p, niter):
        self.cache = {}
        self.cache_hits = 0
        self.G = G
        self.niter = niter 
        self.p = p 
        #Number of evals on per iteration basis
        self.itr_total = 0
        self.itr_new = 0
        self.itr_cache = 0

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
            val = ic_model(self.G, sample, self.p, self.niter)
            self.cache[key] = val
        else:
            self.itr_cache += 1 
            self.cache_hits += 1

        return self.cache[key]

###########################################
def ic_model(G, sample, p, iterations):
#Independent Cascade Model

    N = G.number_of_nodes()
    seed_set = [i for i in range(N) if sample[i] == True]

    avg_influence = Variable(torch.FloatTensor([0]))

    for j in range(iterations):            
        S=list(seed_set)
        for i in range(len(S)):                 # Process each node in seed set
            for neighbor in G.neighbors(S[i]):    
                if random.random()<p:           # Generate a random number and compare it with propagation probability
                    if neighbor not in S:       
                        S.append(neighbor)
        avg_influence = avg_influence + (float(len(S))/iterations) 

    return avg_influence

def main():
    nNodes = 1024 
    p = 0.5
    num_influ_iter = 1
    filename = "/home/pankaj/Sampling/data/input/social_graphs/N_" + str(nNodes) + "/g_N_" + str(nNodes) + "_410.txt"
    G = read_graph(filename, nNodes)
    influ_obj = Influence(G, p, num_influ_iter)
    tic = time.clock()
    for t in range(5000):
        sample = torch.rand(nNodes) > 0.8
        temp = influ_obj(sample.numpy()) 
#        print sample.sum().item(), temp
    print "Average time = ", (time.clock() - tic)/10

if __name__ == '__main__':
#    p_study()
#    variance_study()
#    main()
    variance_study_parallel()
