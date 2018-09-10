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
import argparse
from dpp_objective import DPP 
from variance import variance_estimate

wdir = '/home/pankaj/Sampling/data/working/10_09_2018'
#Reconstruction loss
def reconstruction_loss(p, q):
    #Reconstruction loss - L2 difference between input (p) and proposal (q)
    batch_size = p.size()[0]
    temp = p - q
    l2_norms = torch.norm(temp, 2, 1)
    return ((l2_norms**2).sum())/batch_size

def kl_loss_exact_forward(x, q, dpp):

    kl_value = Variable(torch.Tensor([0]))
    power_set = map(list, itertools.product([0, 1], repeat=dpp.N))
    kl_val_sample = []
    for binary_vec in power_set:

        sample = Variable(torch.from_numpy(np.array(binary_vec)).float())

        f_val = torch.abs(dpp(sample))

        temp = x*sample + (1-x)*(1 - sample)
        prob_x = torch.prod(temp)

        temp = q*sample + (1-q)*(1 - sample)
        prob_q = torch.prod(temp)

        ratio = (f_val*prob_x)/prob_q

        kl_val_sample.append(prob_x*f_val*torch.log(ratio))
        kl_value += prob_x*f_val*torch.log(ratio)

    return sum(kl_val_sample)

#Esimated KL loss
def kl_loss_forward(x_mat, q_mat, dpp, nsamples):

    batch_size = x_mat.size()[0]
    kl_value = torch.Tensor([0])

    kl_value = []
    for p in range(batch_size):

        x = x_mat[p]
        q = q_mat[p]

        f_val = torch.FloatTensor([0]*nsamples)
        N = dpp.N 

        samples = torch.bernoulli(x.repeat(nsamples, 1))

        count = 0

        for t in samples:
            f_val[count] = torch.abs(dpp(t))
            count += 1

        temp = x*samples + (1-x)*(1 - samples)
        prob_x = torch.prod(temp, 1)

        temp = q*samples + (1-q)*(1 - samples)
#        prob_q = torch.prod(temp, 1)
        log_prob_q = torch.log(temp).sum(1)

        kl_value.append(torch.sum(-f_val*log_prob_q))
    
    return sum(kl_value)/(batch_size*nsamples)

#Training function 
#Two phase training - reconstruction loss and then KL loss
def training(x_mat, dpp, args):

    net = MyNet(10)

    net.zero_grad()

    batch_size = int(x_mat.shape[0]) 

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

    #log file
    args_list = [args.recon_lr, args.kl_lr, args.recon_mom, args.kl_mom, args.num_samples_mc]
    file_prefix = wdir + '/dpp_' + '_'.join([str(x) for x in args_list])
    f = open(file_prefix + '_training_log.txt', 'w')

#    optimizer = optim.SGD(net.parameters(), lr=args['recon_lr'], momentum = args['recon_mom'])
    optimizer = optim.Adam(net.parameters(), lr=args.recon_lr)

    start1 = time.time()

    for epoch in range(args.recon_epochs):
        optimizer.zero_grad()   # zero the gradient buffers
        ind = torch.randperm(batch_size)[0:args.minibatch_size]
        minibatch = x_mat[ind]
        output = net(minibatch, adjacency, node_feat, edge_feat) 
        loss = reconstruction_loss(minibatch, output)
        to_print =  [epoch, round(loss.item(), 3), round(time.time() - start1, 1)]
        print "Epoch: ", to_print[0], "       loss (reconstruction) = ", to_print[1] 
        f.write(' '.join([str(x) for x in to_print]) + '\n')
        loss.backward()
        optimizer.step()    # Does the update

    net.zero_grad()

    optimizer2 = optim.SGD(net.parameters(), lr=args.kl_lr, momentum = args.kl_mom)

    for epoch in range(args.kl_epochs):
        optimizer2.zero_grad()   # zero the gradient buffers
        ind = torch.randperm(batch_size)[0:args.minibatch_size]
        minibatch = x_mat[ind]
        output = net(minibatch, adjacency, node_feat, edge_feat) 
        loss = kl_loss_forward(minibatch, output, dpp, args.num_samples_mc)
        loss.backward()
        optimizer2.step()    # Does the update
        if epoch % 20 == 0:
            accurate_loss = kl_loss_forward(minibatch, output, dpp, 1000)
            temp = accurate_loss.item()
        else:
            temp = loss.item()
        to_print =  [epoch, round(temp, 3), round(time.time() - start1, 1)]
        if epoch% 20 == 0:
            print "Epoch: ", to_print[0], "       accurate loss (kl) = ", to_print[1] 
        else:
            print "Epoch: ", to_print[0], "       loss (kl) = ", to_print[1] 

        f.write(' '.join([str(x) for x in to_print]) + '\n')

    torch.save(net.state_dict(), file_prefix + '_net.dat')
    f.close()

    nsamples_list = [1, 5, 10, 20, 50, 100]

    x_copy = x_mat.detach()
    f = open(file_prefix + '_variance.txt', 'w')

    output = net(x_copy, adjacency, node_feat, edge_feat).detach()

    for nsample in nsamples_list:
        no_proposal_var = round(variance_estimate(x_mat, x_mat, dpp, nsample), 3)
        net_proposal_var = round(variance_estimate(x_mat, output, dpp, nsample), 3)
        f.write(str(nsample) + ' ' + str(no_proposal_var) + ' ' + str(net_proposal_var) + '\n')
        print '#samples = ', nsample, ' original variance = ', no_proposal_var, '  variance with learned proposals = ', net_proposal_var 

if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training network using estimated forward KL-based loss for DPPs')
    parser.add_argument('torch_seed', nargs = '?', help='Random seed for torch', type=int, default = 123)
    parser.add_argument('N', nargs = '?', help='# of items in DPP', type=int, default = 100)
    parser.add_argument('recon_lr', nargs = '?', help='Learning rate for reconstruction phase', type=float, default = 1e-3)
    parser.add_argument('kl_lr', nargs = '?', help='Learning rate for KL-based loss minimisation', type=float, default = 1e-2)
    parser.add_argument('recon_mom', nargs = '?', help='Momentum for reconstruction phase', type=float, default = 0.9)
    parser.add_argument('kl_mom', nargs = '?', help='Momentum for KL-based loss phase', type=float, default = 0.9)
    parser.add_argument('recon_epochs', nargs = '?', help='Number of epochs for reconstruction phase', type=float, default = 20)
    parser.add_argument('kl_epochs', nargs = '?', help='Number of epochs for kl-loss phase', type=float, default = 20)

    parser.add_argument('batch_size', nargs = '?', help='Batch size', type=int, default = 100)
    parser.add_argument('minibatch_size', nargs = '?', help='Minibatch size', type=int, default = 10)
    parser.add_argument('num_samples_mc', nargs = '?', help='#samples to use for loss estimation', type=int, default = 100)
    args = parser.parse_args()

    torch.manual_seed(args.torch_seed)

    (qualities, features) = read_dpp('/home/pankaj/Sampling/data/input/dpp/data/clustered_dpp_100_2_200_2_1_5_10.h5', args.N, 'dpp_1') 
    
    dpp = DPP(qualities, features)
 
    x_mat = torch.rand(args.batch_size, args.N)

    training(x_mat, dpp, args)
