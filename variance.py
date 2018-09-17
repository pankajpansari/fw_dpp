import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import torch

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
    return torch.exp(logp_nom - logp_prp)

def getImportanceRelax(x, x_prp, nsamples, dpp): 

    current_sum = torch.FloatTensor([0])
    samples_list = torch.bernoulli(x_prp.repeat(nsamples, 1))
    w = getImportanceWeights(samples_list, x, x_prp)

    for i in range(nsamples):
#        current_sum = current_sum + (w[i]/w.sum())*dpp(samples_list[i])
        current_sum = current_sum + w[i]*dpp(samples_list[i])

    return current_sum/nsamples

def variance_estimate(input, proposal, dpp, nsamples):
    variance_val = []
    batch_size = int(input.size()[0])
    
#    N = int(np.sqrt(int(L_mat[0].shape[0])))
    N = int(input.shape[1])

    for instance in range(batch_size):
        fval = []
        for t in range(1000): #50 seems to work well in practice - smaller (say 20) leads to less consistency of variance
            x = input[instance].unsqueeze(0)
            y = proposal[instance].unsqueeze(0)
            temp = getImportanceRelax(x, y, nsamples, dpp)
            fval.append(temp)
        variance_val.append(np.std(fval)**2)
    return np.mean(variance_val) 

#def variance_study(net_file):
#
#    net = MyNet(10)
#    net.load_state_dict(torch.load(net_file))
#
#    torch.save(net.state_dict(), file_prefix + '_net.dat')
#    nsamples_list = [1, 5, 10, 20, 50, 100]
#
#    (qualities, features)
#    = read_dpp('/home/pankaj/Sampling/data/input/dpp/data/clustered_dpp_100_2_200_2_1_5_10.h5', 100, 'dpp_1') 
#    
#    dpp = DPP(qualities, features)
# 
#    influ_obj = Influence(G, p, num_influ_iter)
#
#    var_list = []
#
#    torch.manual_seed(123)
#
#    x_mat = torch.rand(100, 100)
#    y_mat = net(x_mat)
#    for nsample in nsamples_list:
#        print 

