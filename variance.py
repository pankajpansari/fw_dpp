import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import torch
from read_files import read_dpp
from dpp_objective import DPP 

def getProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.prod(temp, 1)

def getLogProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    a = torch.log(pVec)
    b = torch.log(1 - pVec)
    logp = a * sample + b * (1 - sample)
    return logp.sum(1)

def getImportanceWeights(samples_list, nominal, proposal):
    logp_nom = getLogProb(samples_list, nominal)
    logp_prp = getLogProb(samples_list, proposal)
#    return torch.exp(logp_nom - logp_prp)

    return getProb(samples_list, nominal)/getProb(samples_list, proposal)

def getImportanceRelax(x, x_prp, nsamples, dpp): 

    current_sum = torch.FloatTensor([0])
    samples_list = torch.bernoulli(x_prp.repeat(nsamples, 1))
    w = getImportanceWeights(samples_list, x, x_prp)

    for i in range(nsamples):
#        current_sum = current_sum + (w[i]/w.sum())*dpp(samples_list[i])
        current_sum = current_sum + w[i]*dpp(samples_list[i].detach())
    return current_sum/nsamples

def variance_estimate(input, proposal, dpp, nsamples):
    variance_val = []
    mean_val = []
    batch_size = int(input.size()[0])
    
#    N = int(np.sqrt(int(L_mat[0].shape[0])))
    N = int(input.shape[1])

    for instance in range(batch_size):
        fval = []
        for t in range(1000): #50 seems to work well in practice - smaller (say 20) leads to less consistency of variance
            x = input[instance].unsqueeze(0)
            y = proposal[instance].unsqueeze(0)
            temp = getImportanceRelax(x, y, nsamples, dpp).item()
            fval.append(temp)
        variance_val.append(np.std(fval))
        mean_val.append(np.mean(fval))
    return np.mean(variance_val)


if  __name__ == '__main__':
    torch.manual_seed(123)
    np.random.seed(123)

    (qualities, features) = read_dpp('/home/pankaj/Sampling/data/input/dpp/data/clustered_dpp_20_20_4_3_1_4_1.h5', 'dpp_0')
    dpp = DPP(qualities, features)
 
    N = 20 
    nsamples = 10
    x = torch.rand(N)
    y = x + torch.Tensor(np.random.normal(0, 0.02, N))
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    samples_list = torch.bernoulli(y.repeat(nsamples, 1))
#    print getImportanceWeights(samples_list, x, y)
    print variance_estimate(x, x, dpp, 1)
    print variance_estimate(x, y, dpp, 1)

