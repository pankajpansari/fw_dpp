#!/usr/bin/env python

import sys
import os
import collections
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.autograd import Variable
from graphnet import MyNet
from read_files import read_dpp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import itertools
from tabulate import tabulate
import argparse
from dpp_objective import DPP 
from variance import variance_estimate, getImportanceRelax
from variance_copula import variance_estimate_copula
from bad_grad_viz import register_hooks

wdir = './workspace/'

def write_to_file(f, val_list):
    f.write(' '.join([str(round(x, 3)) for x in val_list]) + '\n')

def print_list(text_list, val_list):
    temp = list(zip(text_list, val_list))
    for (a, b) in temp:
        print a, ': ', str(b), '    ',
    print

def get_copula_prob(u_mat, cov, w):

    N = u_mat.size()[1] 
    Sigma = torch.mm(cov.t(), cov) + w*torch.eye(N)
    normal = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    z = normal.icdf(u_mat)
    Sigma_det = torch.det(Sigma)

    term1 = 1/(torch.sqrt(Sigma_det) + 1e-4)
    inv_mat = torch.inverse(Sigma)
    term3 = inv_mat - torch.eye(N)

    temp = torch.matmul(term3, z.t())
    term2 = torch.mm(z, temp).diag()
    
    nsamples = u_mat.size()[0] 

    if torch.isnan(term2).any():
        print 'term2 has nan'
        sys.exit()
    return term1*torch.exp(-term2/2)

def temp_fn():
    u_mat = torch.rand(2, 20)
    cov = torch.rand((30, 20), requires_grad = True)
    cov.grad = torch.zeros(cov.size())
    prob = get_copula_prob(u_mat, cov, 0)
    prob.sum().backward()
    print cov.grad 
#    get_dot = register_hooks(loss)
#    dot = get_dot()
#    dot.save('tmp.dot')
    sys.exit() 


#Esimated KL loss
def kl_loss_forward(x_mat, marg_mat, cov_mat, dpp, nsamples, w):

    batch_size = x_mat.size()[0]
    kl_value = torch.Tensor([0])
    N = dpp.N 

    kl_value = []
    eps = 1e-4

    for p in range(batch_size):

        x = x_mat[p]
        marg = marg_mat[p]
        cov = cov_mat[p]

        f_val = torch.FloatTensor([0]*nsamples)

        u_mat = torch.empty(nsamples, N).uniform_(0 + 1e-2, 1 - 1e-2)

        samples = (u_mat < x).float()

        copula_prob = get_copula_prob(u_mat, cov, w) 

        count = 0
        for t in samples:
            f_val[count] = torch.abs(dpp(t))
            count += 1

        temp = marg*samples + (1-marg)*(1 - samples)

        log_prob_q = torch.log(temp).sum(1) + torch.log(copula_prob)
        if torch.isnan(temp).any():
            print 'temp has nan'
            sys.exit()

        if torch.isnan(copula_prob).any():
            print 'copula_prob has nan'
            sys.exit()
 
        kl_value.append(torch.sum(-f_val*log_prob_q))
    
    return sum(kl_value)/nsamples

#Training function 
#Two phase training - reconstruction loss and then KL loss
def training(x_mat, dpp, args):

    net = MyNet()

    net.zero_grad()

    batch_size = int(x_mat.shape[0]) 

    #Quality and feature vector as node_feat
    node_feat = 100*torch.unsqueeze(dpp.qualities, 0)

    #Concatenated feature vectors and qualities + dot product
    edge_feat = torch.zeros(1, dpp.N, dpp.N)

    for i in range(dpp.N):
        for j in range(dpp.N):
            feat_i = dpp.features[:, i]
            feat_j = dpp.features[:, j]
            quality_term = (dpp.qualities[i]*dpp.qualities[j])**2
            diversity_term = 1 - (feat_i.dot(feat_j))**2
            edge_feat[0, i, j] = 100*quality_term * diversity_term 

    #Fully-connected graph with diagonal elements 0
    adjacency = torch.ones(dpp.N, dpp.N) 
    idx = torch.arange(0, dpp.N, out = torch.LongTensor())
    adjacency[idx, idx] = 0

    #log file
    args_list = [args.torch_seed, args.dpp_id, args.N, args.kl_lr, args.kl_mom, args.kl_epochs, args.batch_size, args.minibatch_size, args.num_samples_mc]
    file_prefix = wdir + '/dpp_' + '_'.join([str(x) for x in args_list])

    f = open(file_prefix + '_stochastic_training_log.txt', 'w', 0)

    start1 = time.time()

    optimizer = optim.RMSprop(net.parameters(), lr=args.kl_lr, momentum = args.kl_mom, eps = 1e-4)

    net.zero_grad()
#    for name, param in net.named_parameters():
#        print name, param.grad
#    sys.exit() 

    w = 1
    for epoch in range(args.kl_epochs):
        optimizer.zero_grad()   # zero the gradient buffers
        ind = torch.randperm(batch_size)[0:args.minibatch_size]
        minibatch = x_mat[ind]
        [marg, cov] = net(minibatch, adjacency, node_feat, edge_feat) 
        w = 0.99*w
        loss = kl_loss_forward(minibatch, marg, cov, dpp, args.num_samples_mc, w)
        loss.backward()
        optimizer.step()    # Does the update
 
        avg_loss = loss/args.minibatch_size
        text_list = ['Epoch', 'Loss']

        val_list =  [epoch, round(avg_loss.item(), 3), round(time.time() - start1, 1)]

        print_list(text_list, val_list)

        write_to_file(f, val_list)

    f.close()
    sys.exit()
    torch.save(net.state_dict(), file_prefix + '_net.dat')

#    temp = torch.load(file_prefix + '_net.dat')
#
#    net.load_state_dict(temp)

    testing(net, x_mat, dpp, file_prefix + '_train_variance.txt')


def testing(net, x_mat, dpp, filename):

    #Quality and feature vector as node_feat
    node_feat = torch.unsqueeze(dpp.qualities, 0)

    #Concatenated feature vectors and qualities + dot product
    edge_feat = torch.zeros(1, dpp.N, dpp.N)

    for i in range(dpp.N):
        for j in range(dpp.N):
            feat_i = dpp.features[:, i]
            feat_j = dpp.features[:, j]
            quality_term = (dpp.qualities[i]*dpp.qualities[j])**2
            diversity_term = 1 - (feat_i.dot(feat_j))**2
            edge_feat[0, i, j] = quality_term * diversity_term 

    #Fully-connected graph with diagonal elements 0
    adjacency = torch.ones(dpp.N, dpp.N) 
    idx = torch.arange(0, dpp.N, out = torch.LongTensor())
    adjacency[idx, idx] = 0

    nsamples_list = [1, 5]

    f = open(filename, 'w')

    [marg, cov] = net(x_mat, adjacency, node_feat, edge_feat)
    
    print x_mat, marg
    print cov
    for nsample in nsamples_list:
        no_proposal_var = round(variance_estimate(x_mat, x_mat, dpp, nsample), 3)
        net_proposal_var = round(variance_estimate_copula(x_mat, marg, cov, dpp, nsample), 3)
        param_list = [nsample, no_proposal_var, net_proposal_var]
        text_list = ['#samples', 'original variance', 'variance with learned proposals']
        write_to_file(f, param_list)
        print_list(text_list, param_list)

    f.close()



if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training network using estimated forward KL-based loss for DPPs')
    parser.add_argument('torch_seed', nargs = '?', help='Random seed for torch', type=int, default = 123)
    parser.add_argument('N', nargs = '?', help='# of items in DPP', type=int, default = 20)
    parser.add_argument('dpp_id', nargs = '?', help='id of DPP', type=int, default = 0)
    parser.add_argument('num_samples_mc', nargs = '?', help='#samples to use for loss estimation', type=int, default = 1000)
    parser.add_argument('kl_lr', nargs = '?', help='Learning rate for KL-based loss minimisation', type=float, default = 1e-4)
    parser.add_argument('kl_mom', nargs = '?', help='Momentum for KL-based loss phase', type=float, default = 0.9)
    parser.add_argument('kl_epochs', nargs = '?', help='Number of epochs for kl-loss phase', type=int, default = 3000)
    parser.add_argument('batch_size', nargs = '?', help='Batch size', type=int, default = 1)
    parser.add_argument('minibatch_size', nargs = '?', help='Minibatch size', type=int, default = 1)

    args = parser.parse_args()
    torch.manual_seed(args.torch_seed)

    (qualities, features) = read_dpp('/home/pankaj/Sampling/data/input/dpp/data/clustered_dpp_20_20_4_3_1_4_1.h5', 'dpp_' + str(args.dpp_id))

    dpp = DPP(qualities, features)
 
    y_mat = torch.Tensor(np.reshape(np.loadtxt('/home/pankaj/Sampling/code/fw_dpp/workspace/dpp_123_0_20_10_1_100_fw_simple_iterates.txt'), (100, args.N)))
    x_mat = y_mat[0:args.batch_size, :]
    x_mat = torch.rand(args.batch_size, args.N)
    training(x_mat, dpp, args)

    sys.exit()
#    x_val_mat = torch.rand(args.batch_size, args.N)

    net = MyNet()
    args_list = [args.torch_seed, args.dpp_id, args.N, args.kl_lr, args.kl_mom, args.kl_epochs, args.batch_size, args.minibatch_size, args.num_samples_mc]

    file_prefix = wdir + '/dpp_' + '_'.join([str(x) for x in args_list])
#    temp = torch.load(file_prefix + '_net.dat')
    temp = torch.load('workspace/dpp_125_0_20_0.0001_0.9_2000_10_10_1000_net.dat')
    net.load_state_dict(temp)
    testing(net, x_mat, dpp, file_prefix + '_test_variance.txt')

