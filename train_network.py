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
        prob_x = torch.prod(temp, 1)

        temp = q*sample + (1-q)*(1 - sample)
        prob_q = torch.prod(temp, 1)

        ratio = (f_val*prob_x)/prob_q

        kl_val_sample.append(prob_x*f_val*torch.log(ratio))
        kl_value += prob_x*f_val*torch.log(ratio)

    return sum(kl_val_sample)

#Esimated KL loss
def kl_loss_forward(x_mat, q_mat, dpp, nsamples):

    batch_size = x_mat.size()[0]
    kl_value = torch.Tensor([0])

    for p in range(batch_size):

        x = x_mat[p]
        q = q_mat[p]

        f_val = Variable(torch.FloatTensor([0]*nsamples)) 
        N = dpp.N 

        samples = Variable(torch.bernoulli(x.repeat(nsamples, 1)))

        count = 0

        for t in samples:
            f_val[count] = torch.abs(dpp(t))
            count += 1

        temp = x*samples + (1-x)*(1 - samples)
        prob_x = torch.prod(temp, 1)

        temp = q*samples + (1-q)*(1 - samples)
        prob_q = torch.prod(temp, 1)

        ratio = (f_val*prob_x)/prob_q

        kl_value += torch.sum(f_val*torch.log(ratio))/nsamples
    
    return kl_value/batch_size


def kl_loss_exact_reverse(x, q, dpp):

    kl_value = Variable(torch.Tensor([0]))
    power_set = map(list, itertools.product([0, 1], repeat=dpp.N))

    #get normalisation constant
    C = Variable(torch.Tensor([0]))

    sample_loss = []
    for binary_vec in power_set:

        sample = Variable(torch.from_numpy(np.array(binary_vec)).float())

        f_val = torch.abs(dpp(sample))

        temp = x*sample + (1-x)*(1 - sample)
        prob_x = torch.prod(temp, 1)

        temp = q*sample + (1-q)*(1 - sample)
        prob_q = torch.prod(temp, 1)

        ratio = prob_q/(f_val*prob_x)

        sample_loss.append(prob_q*torch.log(ratio))

        kl_value += prob_q*torch.log(ratio)

        C = torch.add(C, f_val*prob_x)

    exact_kl_value = kl_value + torch.log(C) 
    return sum(sample_loss)
#    return kl_value
    


def kl_loss_reverse(x_mat, q_mat, dpp, nsamples):

    batch_size = x_mat.size()[0]
    kl_value = torch.Tensor([0])

    rewards = []
    for p in range(batch_size):

        x = x_mat[p]
        q = q_mat[p]

        f_val = Variable(torch.FloatTensor([0]*nsamples)) 
        N = dpp.N 

        samples = Variable(torch.bernoulli(q.repeat(nsamples, 1)))

        count = 0

        for t in samples:
            f_val[count] = torch.abs(dpp(t.numpy()))
            count += 1

        temp = x*samples + (1-x)*(1 - samples)
        prob_x = torch.prod(temp, 1)

        temp = q*samples + (1-q)*(1 - samples)
        prob_q = torch.prod(temp, 1)

        ratio = prob_q/(f_val*prob_x)

        importance_weights = prob_q/prob_x

#        kl_value += torch.sum(importance_weights * torch.log(ratio))/nsamples
        kl_value += torch.sum(torch.log(ratio))/nsamples
    
    return kl_value/batch_size

def kl_loss_reverse2(x_mat, q_mat, dpp, nsamples):

    kl_value = Variable(torch.Tensor([0]))

    reward = []

    x = x_mat[0, :]
    q = q_mat[0, :]

    for t in range(nsamples):

        sample = torch.bernoulli(q)

        f_val = torch.abs(dpp(sample))

        temp = x*sample + (1-x)*(1 - sample)
        prob_x = torch.prod(temp)

        temp = q*sample + (1-q)*(1 - sample)
        prob_q = torch.prod(temp)

        ratio = prob_q/(f_val*prob_x)

        reward.insert(0, -torch.log(ratio))

    return -sum(reward)/nsamples


def reinforce_func(x, q, dpp, sample):
    f_val = torch.abs(dpp(sample.numpy()))

    temp = x*sample + (1-x)*(1 - sample)
    prob_x = torch.prod(temp, 1)

    temp = q*sample + (1-q)*(1 - sample)
    prob_q = torch.prod(temp, 1)

def reinforce_grad(x, q, dpp, nsamples):
    samples = Variable(torch.bernoulli(q.repeat(nsamples, 1)))
    for sample in samples:
        temp = reinforce_func(x, q, dpp, sample) 
        temp.backward()
    print q.grad.data/nsamples        

#Training function 
#Two phase training - reconstruction loss and then KL loss
def training(x_mat, dpp, args):

    net = MyNet(10)

    net.zero_grad()
#    optimizer = optim.SGD(net.parameters(), lr=args['recon_lr'], momentum = args['recon_mom'])
    optimizer = optim.Adam(net.parameters(), lr=args['recon_lr'])

    batch_size = int(x_mat.shape[0]) 

    #Quality and feature vector as node_feat
    node_feat = torch.cat((torch.unsqueeze(dpp.qualities, 0), dpp.features), 0)
    node_feat = node_feat.t()
    node_feat = node_feat.repeat(batch_size, 1, 1) 

    #Concatenated feature vectors and qualities + dot product
    edge_feat = torch.zeros(1, dpp.N, dpp.N)

    #Fully-connected graph
    adjacency = torch.ones(dpp.N, dpp.N) 
    adjacency = adjacency.repeat(batch_size, 1, 1)

    for epoch in range(args['recon_epochs']):
        optimizer.zero_grad()   # zero the gradient buffers
#        ind = torch.randperm(batch_size)[0:args['minibatch_size']]
#        minibatch = x_mat[ind]
        minibatch = x_mat
        output = net(minibatch, adjacency, node_feat, edge_feat) 
        loss = reconstruction_loss(minibatch, output)
        print "Epoch: ", epoch, "       loss (l2 reconstruction) = ", loss.data.numpy()
        loss.backward()
        optimizer.step()    # Does the update

    net.zero_grad()

    optimizer2 = optim.SGD(net.parameters(), lr=args['kl_lr'], momentum = args['kl_mom'])

    for epoch in range(args['kl_epochs']):
        optimizer2.zero_grad()   # zero the gradient buffers
#        ind = torch.randperm(batch_size)[0:args['minibatch_size']]
#        minibatch = x_mat[ind]
        minibatch = x_mat
        output = net(minibatch, adjacency, node_feat, edge_feat) 
        loss = kl_loss_forward(minibatch, output, dpp, args['num_samples_mc'])
#        loss = kl_loss_exact(minibatch, output, dpp)
        loss.backward()
        optimizer2.step()    # Does the update
        print output
#        print loss.grad_fn
#        print [x.grad.data for x in net.parameters()]
        print "Epoch: ", epoch, "       loss (kl) = ", loss.item()

#    print x_mat
#    print net(x_mat, adjacency, node_feat, edge_feat)
#main function accepting as input:
#-DPP matrices
#-training parameters

if  __name__ == '__main__':

    N = 10 

    (qualities, features) = read_dpp('/home/pankaj/Sampling/data/input/dpp/data/clustered_dpp_10_2_20_2_1_5_2.h5', N, 'dpp_1') 
    
    dpp = DPP(qualities, features)

    x = torch.rand(1, N)
    y = torch.rand(1, N, requires_grad = True)

    loss = kl_loss_exact_forward(x, y, dpp)
    print loss
    loss.backward() 
    print y.grad.data

    y.grad.data = torch.zeros(y.size())
    loss = kl_loss_forward(x, y, dpp, 1000)
    print loss
    loss.backward()
    print y.grad.data


#if  __name__ == '__main__':
#    N = 100
#    (qualities, features) = read_dpp('/home/pankaj/Sampling/data/input/dpp/data/clustered_dpp_100_2_200_2_1_5_10.h5', N, 'dpp_1') 
#    
#    dpp = DPP(qualities, features)
# 
#    x_mat = torch.rand(2, N)
#
#    args = {'recon_lr': 1e-3,  'kl_lr' : 1e-6,  'recon_mom' : 1e-3,  'kl_mom' : 1e-5, 'recon_epochs' : 500, 'kl_epochs' : 4, 'minibatch_size' : 1,  'num_samples_mc' : 1}
#    training(x_mat, dpp, args)
