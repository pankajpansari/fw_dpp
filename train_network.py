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

torch.manual_seed(123)
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
#    optimizer = optim.SGD(net.parameters(), lr=args['recon_lr'], momentum = args['recon_mom'])
    optimizer = optim.Adam(net.parameters(), lr=args['recon_lr'])

    batch_size = int(x_mat.shape[0]) 

    #Quality and feature vector as node_feat
    node_feat = torch.unsqueeze(dpp.qualities, 0)
    node_feat = node_feat.repeat(batch_size, 1, 1) 

    #Concatenated feature vectors and qualities + dot product
    edge_feat = torch.zeros(1, dpp.N, dpp.N)

    for i in range(dpp.N):
        for j in range(dpp.N):
            feat_i = dpp.features[:, i]
            feat_j = dpp.features[:, j]
            quality_term = (dpp.qualities[i]*dpp.qualities[j])**2
            diversity_term = 1 - (feat_i.dot(feat_j))**2
            edge_feat[0, i, j] = quality_term * diversity_term 

    edge_feat = edge_feat.repeat(batch_size, 1, 1, 1) 

    #Fully-connected graph with diagonal elements 0
    adjacency = torch.ones(dpp.N, dpp.N) 
    idx = torch.arange(0, dpp.N, out = torch.LongTensor())
    adjacency[idx, idx] = 0
    adjacency = adjacency.repeat(batch_size, 1, 1)

    for epoch in range(args['recon_epochs']):
        optimizer.zero_grad()   # zero the gradient buffers
        ind = torch.randperm(batch_size)[0:args['minibatch_size']]
        minibatch = x_mat[ind]
        output = net(minibatch, adjacency, node_feat, edge_feat) 
        loss = reconstruction_loss(minibatch, output)
        print "Epoch: ", epoch, "       loss (l2 reconstruction) = ", loss.data.numpy()
        loss.backward()
        optimizer.step()    # Does the update

    net.zero_grad()

    optimizer2 = optim.SGD(net.parameters(), lr=args['kl_lr'], momentum = args['kl_mom'])

    for epoch in range(args['kl_epochs']):
        optimizer2.zero_grad()   # zero the gradient buffers
        ind = torch.randperm(batch_size)[0:args['minibatch_size']]
        minibatch = x_mat[ind]
        output = net(minibatch, adjacency, node_feat, edge_feat) 
        loss = kl_loss_forward(minibatch, output, dpp, args['num_samples_mc'])
        loss.backward()
#        for params in net.parameters():
#            print params.grad.data
#            if (params.grad == float('inf')).sum() >= 1:
#                print params.grad
#                sys.exit()
        optimizer2.step()    # Does the update
        if epoch % 20 == 0:
            accurate_loss = kl_loss_forward(minibatch, output, dpp, 10000)
            print "Epoch: ", epoch, "       accurate loss (kl) = ", accurate_loss.item()
        else:
            print "Epoch: ", epoch, "       loss (kl) = ", loss.item()

if  __name__ == '__main__':

    N = 100
#    (qualities, features) = read_dpp('/home/pankaj/Sampling/data/input/dpp/data/clustered_dpp_10_2_20_2_1_5_2.h5', N, 'dpp_1') 
    (qualities, features) = read_dpp('/home/pankaj/Sampling/data/input/dpp/data/clustered_dpp_100_2_200_2_1_5_10.h5', N, 'dpp_1') 
    
    dpp = DPP(qualities, features)
 
    x_mat = torch.rand(1, N)

    args = {'recon_lr': 1e-3,  'kl_lr' : 1e-2,  'recon_mom' : 1e-3,  'kl_mom' : 0.9, 'recon_epochs' : 100, 'kl_epochs' : 1000, 'minibatch_size' : 1,  'num_samples_mc' : 100}
    training(x_mat, dpp, args)
