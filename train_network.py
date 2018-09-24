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

wdir = './workspace/'

def write_to_file(f, val_list):
    f.write(' '.join([str(round(x, 3)) for x in val_list]) + '\n')

def print_list(text_list, val_list):
    temp = list(zip(text_list, val_list))
    for (a, b) in temp:
        print a, ': ', str(b), '    ',
    print

#Reconstruction loss
def reconstruction_loss(p, q):
    #Reconstruction loss - L2 difference between input (p) and proposal (q)
    batch_size = p.size()[0]
    temp = p - q
    l2_norms = torch.norm(temp, 2, 1)
    return ((l2_norms**2).sum())

def kl_loss_exact_reverse(x_mat, q_mat, dpp):
    power_set = map(list, itertools.product([0, 1], repeat= x_mat.shape[1]))

    batch_size = x_mat.size()[0]

    kl_mat = []

    for p in range(batch_size):

        x = x_mat[p]
        q = q_mat[p]

        C = torch.Tensor([0])

        kl_val = torch.Tensor([0])

        for binary_vec in power_set:

            sample = Variable(torch.from_numpy(np.array(binary_vec)).float())

            f_val = torch.abs(dpp(sample))

            temp = x*sample + (1-x)*(1 - sample)
            log_prob_x = torch.log(temp).sum()
            prob_x = torch.prod(temp)

            temp = q*sample + (1-q)*(1 - sample)
            log_prob_q = torch.log(temp).sum()
            prob_q = torch.prod(temp)

            log_term = log_prob_q - torch.log(f_val) - log_prob_x 

            C = torch.add(C, f_val*prob_x)
            kl_val = torch.add(kl_val, prob_q*log_term)
        kl_mat.append(kl_val + torch.log(C))

    return sum(kl_mat)


def kl_loss_exact_forward(x_mat, q_mat, dpp):

    power_set = map(list, itertools.product([0, 1], repeat= x_mat.shape[1]))

    batch_size = x_mat.size()[0]

    kl_mat = []

    epsilon = 1e-5

    for p in range(batch_size):

        x = x_mat[p]
        q = q_mat[p]

        C = torch.Tensor([0])

        kl_val = torch.Tensor([0])

        for binary_vec in power_set:

            sample = Variable(torch.from_numpy(np.array(binary_vec)).float())

            f_val = torch.abs(dpp(sample))

            temp = x*sample + (1-x)*(1 - sample)
            log_prob_x = torch.log(temp).sum()
            prob_x = torch.prod(temp)

            temp = q*sample + (1-q)*(1 - sample)
            log_prob_q = torch.log(temp).sum()

            log_term = torch.log(f_val + epsilon) + log_prob_x - log_prob_q

            C = torch.add(C, f_val*prob_x)
            kl_val = torch.add(kl_val, prob_x*f_val*log_term)

        kl_mat.append(kl_val/C - torch.log(C))
    return sum(kl_mat)

def getConstant(x, q, dpp):

    power_set = map(list, itertools.product([0, 1], repeat= x_mat.shape[1]))

    C = torch.Tensor([0])

    for binary_vec in power_set:

        sample = Variable(torch.from_numpy(np.array(binary_vec)).float())

        f_val = torch.abs(dpp(sample))

        temp = x*sample + (1-x)*(1 - sample)
        prob_x = torch.prod(temp)

        C = torch.add(C, f_val*prob_x)

    return C 

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
    
    return sum(kl_value)/(nsamples)

#Training function 
#Two phase training - reconstruction loss and then KL loss
def training(x_mat, dpp, args):

    net = MyNet()

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
    args_list = [args.torch_seed, args.dpp_id, args.N, args.kl_lr, args.kl_mom, args.kl_epochs, args.batch_size, args.minibatch_size, args.num_samples_mc]
    file_prefix = wdir + '/dpp_' + '_'.join([str(x) for x in args_list])

    f = open(file_prefix + '_stochastic_training_log.txt', 'w')

    start1 = time.time()

    net.zero_grad()

    optimizer = optim.Adam(net.parameters(), lr=args.kl_lr)
    optimizer = optim.RMSprop(net.parameters(), lr=args.kl_lr, momentum = args.kl_mom, eps = 1e-4)

#    for epoch in range(2):
    for epoch in range(args.kl_epochs):
        optimizer.zero_grad()   # zero the gradient buffers
        ind = torch.randperm(batch_size)[0:args.minibatch_size]
        minibatch = x_mat[ind]
        output = net(minibatch, adjacency, node_feat, edge_feat) 
        loss = kl_loss_forward(minibatch, output, dpp, args.num_samples_mc)
#        loss = kl_loss_exact_forward(minibatch, output, dpp)
        loss.backward()
        optimizer.step()    # Does the update

        if epoch % 20 == 0:
            accurate_loss = kl_loss_forward(minibatch, output, dpp, 100)
            avg_loss = loss/args.minibatch_size
            text_list = ['Epoch', 'Accurate loss']
        else:
            avg_loss = loss/args.minibatch_size
            text_list = ['Epoch', 'Loss']

        val_list =  [epoch, round(avg_loss.item(), 3), round(time.time() - start1, 1)]

        print_list(text_list, val_list)

        write_to_file(f, val_list)

    f.close()

    torch.save(net.state_dict(), file_prefix + '_net.dat')

#    temp = torch.load(file_prefix + '_net.dat')
#    net.load_state_dict(temp)

#    output = net(x_mat, adjacency, node_feat, edge_feat) 
#    x = x_mat[0]
#    q = output[0]
#    C = getConstant(x, q, dpp)
#    table = []
#
#    for t in range(10):
#        sample = torch.bernoulli(torch.Tensor([0.5]*args.N))
#        f_val = torch.abs(dpp(sample))
#
#        temp = x*sample + (1-x)*(1 - sample)
#        prob_x = torch.prod(temp)
#
#        temp = q*sample + (1-q)*(1 - sample)
#        prob_q = torch.prod(temp)
#
#        table.append(['Sample ' + str(t),
#            '{:0.2e}'.format((f_val*prob_x/C).item()),
#            '{:0.2e}'.format((prob_q).item())])
#        print (f_val*prob_x/C).item(), (prob_q).item()
#    print tabulate(table, headers = ['Sample', 'Optimal probability', 'Learned probability'], tablefmt = 'latex') 

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

    x_copy = x_mat.detach()
    f = open(filename, 'w')

    output = net(x_copy, adjacency, node_feat, edge_feat).detach()

    for nsample in nsamples_list:
        no_proposal_var = round(variance_estimate(x_mat, x_mat, dpp, nsample), 3)
        net_proposal_var = round(variance_estimate(x_mat, output, dpp, nsample), 3)
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

    (qualities, features) = read_dpp('/home/pankaj/Sampling/data/input/dpp/data/clustered_dpp_100_2_200_1_2_5_20.h5', 'dpp_' + str(args.dpp_id))

    dpp = DPP(qualities, features)
 
#    x_mat = torch.rand(args.batch_size, args.N)
    x_mat = torch.Tensor(np.reshape(np.loadtxt('/home/pankaj/Sampling/code/fw_dpp/workspace/dpp_123_0_100_20_100_100_fw_simple_iterates.txt'), (args.batch_size, args.N)))

#    nsamples_list = [1, 5, 10]
#
#    x_copy = x_mat.detach()
#
#    for nsample in nsamples_list:
#        no_proposal_var = round(variance_estimate(x_mat, x_mat, dpp, nsample), 3)
#        param_list = [nsample, no_proposal_var]
#        text_list = ['#samples', 'original variance']
#        print_list(text_list, param_list)


#    sys.exit()
    training(x_mat, dpp, args)

    sys.exit()
    x_val_mat = torch.rand(args.batch_size, args.N)

    net = MyNet()
    args_list = [args.torch_seed, args.dpp_id, args.N, args.kl_lr, args.kl_mom, args.kl_epochs, args.batch_size, args.minibatch_size, args.num_samples_mc]
#    args_list = [args.torch_seed, args.dpp_id, args.N, args.recon_lr, args.kl_lr, args.recon_mom, args.kl_mom, args.recon_epochs, args.kl_epochs, args.batch_size, args.minibatch_size, args.num_samples_mc]
    file_prefix = wdir + '/dpp_' + '_'.join([str(x) for x in args_list])
    temp = torch.load(file_prefix + '_net.dat')
    net.load_state_dict(temp)
    testing(net, x_val_mat, dpp, file_prefix + '_test_variance.txt')

