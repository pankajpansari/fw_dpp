import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import torch
import networkx as nx
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import itertools
from objective import submodObj
from read_files import get_sfo_optimum, get_fw_optimum, read_graph, read_iterates
from influence import Influence 
from frank_wolfe_importance import getImportanceRelax
from frank_wolfe import getRelax

def getProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.prod(temp, 1)

def getLogProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.sum(torch.log(temp), 1)


def multilinear_importance(x, y, graph_file, nsamples): #x is targe probability vector and y is the proposal distribution
    running_sum = Variable(torch.FloatTensor([0]), requires_grad = True) 
    for trial in range(nsamples):
        sample = Variable(torch.bernoulli(y[0].data))
        targetP = getProb(sample, x)
        proposalP = getProb(sample, y)
        assert proposalP.data[0] > 0, "Proposal distribution invalid"
        running_sum = running_sum + submodObj(graph_file, sample)*(float(targetP)/proposalP)
    mc_val = running_sum/nsamples
    return mc_val.data[0]


def variance_estimate(input, proposal, graph_file, nsamples):
    variance_val = []
    batch_size = int(input.size()[0])
    
#    N = int(np.sqrt(int(L_mat[0].shape[0])))
    N = int(input.shape[1])

    for instance in range(batch_size):
        fval = []
        for t in range(50): #50 seems to work well in practice - smaller (say 20) leads to less consistency of variance
            x = input[instance].unsqueeze(0)
            y = proposal[instance].unsqueeze(0)
            temp = multilinear_importance(x, y, graph_file, nsamples)
            fval.append(temp)
        variance_val.append(np.std(fval)**2)
    return np.median(variance_val), np.mean(variance_val) 

def variance_study(G, nsamples, k, var_file, p, num_influ_iter, if_herd, x_good_sfo, x_good_fw, x, a):

    N = nx.number_of_nodes(G)

    influ_obj = Influence(G, p, num_influ_iter)

    temp = []

    if a == 0:
        for t in range(40):
            val = getRelax(G, x, nsamples, influ_obj, if_herd).item()
            temp.append((val, val, val))

    else:
        for t in range(40):
            val1 = getImportanceRelax(G, x_good_sfo, x, nsamples, influ_obj, if_herd, a).item()
            val2 = getImportanceRelax(G, x_good_fw, x, nsamples, influ_obj, if_herd, a).item()
            val3 = getRelax(G, x, nsamples, influ_obj, if_herd).item()
            temp.append((val1, val2, val3))

    relax_gt = getRelax(G, x, 200, influ_obj, if_herd).item()

    print('\n'*2)
    print("sfo std= ", np.std([t[0] for t in temp]), "  mean = ", np.mean([t[0] for t in temp]))
    print("fw std = ", np.std([t[1] for t in temp]), "  mean = ", np.mean([t[1] for t in temp]))
    print("mc std = ", np.std([t[2] for t in temp]), "  mean = ", np.mean([t[2] for t in temp]))
    print("gt = ", relax_gt)

    f = open(var_file, 'a', 0)
    f.write(str(np.std([t[0] for t in temp]))+ " " + str(np.mean([t[0] for t in temp])) + " " + str(relax_gt) + "\n")
    f.write(str(np.std([t[1] for t in temp]))+ " " + str(np.mean([t[1] for t in temp])) + " " + str(relax_gt) + "\n")
    f.write(str(np.std([t[2] for t in temp]))+ " " + str(np.mean([t[2] for t in temp])) + " " + str(relax_gt) + "\n")
    f.write('\n')
    f.close()

def convex_var(N, g_id, k, nsamples, p, num_influ_iter, if_herd, a):

    graph_file = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/graphs/g_N_' + str(N) + '_' + str(g_id) + '.txt'

    G = read_graph(graph_file, N)

    x_good_sfo = get_sfo_optimum('/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/sfo_gt/g_N_' + str(N) + '_id_' + str(g_id) + '_k_' + str(k) + '.txt', N) 

    x_good_fw = get_fw_optimum('/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/fw_gt/g_N_' + str(N) + '_id_' + str(g_id) + '_k_' + str(k) + '_100.txt', N) 

    num_iterates = 10 

    x_list = read_iterates('/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/iterates/g_N_' + str(N) + '_' + str(g_id) + '_' + str(k) + '_100_10_0.4_100_0_0_0.txt', N, num_iterates)

    temp = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/var_study/g_N_' + str(N) + '_' + str(g_id) 

    var_file = '_'.join(str(y) for y in [temp, k, nsamples, p, num_influ_iter, if_herd, a]) + '.txt'

    #Empty file contents
    f = open(var_file, 'w', 0)
    f.close()

    for x in x_list:
        variance_study(G, nsamples, k, var_file, p, num_influ_iter, if_herd, x_good_sfo, x_good_fw, x, a) 
