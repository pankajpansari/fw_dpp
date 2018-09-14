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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import itertools
import argparse
from variance import variance_estimate

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

def set_fn(mod_obj, sample):
    a = (1- sample).type(torch.ByteTensor)
    b = sample.type(torch.ByteTensor)
    return torch.prod(mod_obj[0, a])*torch.prod(mod_obj[1, b])

def optimal_proposal(mod_obj, x):
    q = x*mod_obj[1]/(x*mod_obj[1] + (1 - x)*mod_obj[0])
    return q

def kl_loss_reverse(x_mat, q_mat, mod_obj, nsamples):

    batch_size = x_mat.size()[0]
    N = x_mat.size()[1]
    kl_value = [] 

    for p in range(batch_size):

        x = x_mat[p]
        q = q_mat[p]

        f_val = Variable(torch.FloatTensor([0]*nsamples)) 

        samples = Variable(torch.bernoulli(x.repeat(nsamples, 1)))

        count = 0

        for t in samples:
            f_val[count] = torch.abs(set_fn(mod_obj, t))
            count += 1

        temp = x*samples + (1-x)*(1 - samples)
        log_prob_x = torch.log(temp).sum(1)
        prob_x = torch.prod(temp, 1)

        temp = q*samples + (1-q)*(1 - samples)
        log_prob_q = torch.log(temp).sum(1)
        prob_q = torch.prod(temp, 1)

        ratio = prob_q/(f_val*prob_x)

        log_term = log_prob_q - torch.log(f_val) - log_prob_x  

        importance_weights = prob_q/prob_x

        kl_value.append(torch.sum(importance_weights*log_term))
    
    return sum(kl_value)/(nsamples)

#Esimated KL loss
def kl_loss_forward(x_mat, q_mat, mod_obj,  nsamples):

    batch_size = x_mat.size()[0]

    kl_value = []

    for p in range(batch_size):

        x = x_mat[p]
        q = q_mat[p]

        f_val = torch.FloatTensor([0]*nsamples)
        N = mod_obj.shape[0] 

        samples = torch.bernoulli(x.repeat(nsamples, 1))

        count = 0

        for t in samples:
            f_val[count] = torch.abs(set_fn(mod_obj, t))
            count += 1

        temp = x*samples + (1-x)*(1 - samples)
        prob_x = torch.prod(temp, 1)

        temp = q*samples + (1-q)*(1 - samples)
        log_prob_q = torch.log(temp).sum(1)

        kl_value.append(torch.sum(-f_val*log_prob_q))
    
    return sum(kl_value)/(nsamples)

def kl_loss_exact_forward(x_mat, q_mat, mod_obj):

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

            f_val = torch.abs(set_fn(mod_obj, sample))

            temp = x*sample + (1-x)*(1 - sample)
            log_prob_x = torch.log(temp).sum()
            prob_x = torch.prod(temp)

            temp = q*sample + (1-q)*(1 - sample)
            log_prob_q = torch.log(temp).sum()

            log_term = torch.log(f_val) + log_prob_x - log_prob_q

            C = torch.add(C, f_val*prob_x)
            kl_val = torch.add(kl_val, prob_x*f_val*log_term)
        kl_mat.append(kl_val/C - torch.log(C))
    return sum(kl_mat)

#Training function 
#Two phase training - reconstruction loss and then KL loss
def training(x_mat, mod_obj, args):

    net = MyNet(args.k)

    net.zero_grad()

    batch_size = int(x_mat.shape[0]) 

    #Quality and feature vector as node_feat
    node_feat = mod_obj
    
    #Concatenated feature vectors and qualities + dot product
    edge_feat = torch.zeros(1, args.N, args.N)

    for i in range(args.N):
        for j in range(args.N):
            sample = torch.zeros(args.N)
            sample[i] = 1
            sample[j] = 1
            edge_feat[0, i, j] = set_fn(mod_obj, sample)

    #Fully-connected graph with diagonal elements 0
    adjacency = torch.ones(args.N, args.N) 
    idx = torch.arange(0, args.N, out = torch.LongTensor())
    adjacency[idx, idx] = 0

    #log file
    args_list = [args.recon_lr, args.kl_lr, args.recon_mom, args.kl_mom, args.recon_epochs, args.kl_epochs, args.batch_size, args.minibatch_size, args.num_samples_mc]
    file_prefix = wdir + '/mod_' + '_'.join([str(x) for x in args_list])
    f = open(file_prefix + '_training_log.txt', 'w')

#    optimizer = optim.SGD(net.parameters(), lr=args.recon_lr, momentum = args.recon_mom)
    optimizer = optim.Adam(net.parameters(), lr=args.recon_lr)

    start1 = time.time()

    output = net(x_mat, adjacency, node_feat, edge_feat) 
    y_mat = torch.rand(x_mat.size())
    q_opt = optimal_proposal(mod_obj, x_mat)
    l2_dist_initial = reconstruction_loss(q_opt, output)/args.minibatch_size    
#    print q_opt[0] 
#    print y_mat[0]
#    print ((q_opt[0] - y_mat[0])**2).sum()
#    sys.exit()
    for epoch in range(args.recon_epochs):
        optimizer.zero_grad()   # zero the gradient buffers
        ind = torch.randperm(batch_size)[0:args.minibatch_size]
        minibatch = x_mat[ind]
        output = net(minibatch, adjacency, node_feat, edge_feat) 
        loss = reconstruction_loss(minibatch, output)
        avg_loss = loss.detach().sum()/args.minibatch_size
        to_print =  [epoch, round(avg_loss.item(), 3), round(time.time() - start1, 1)]
        print "Epoch: ", to_print[0], "       loss (reconstruction) = ", to_print[1] 
        f.write(' '.join([str(x) for x in to_print]) + '\n')
        loss.backward()
        optimizer.step()    # Does the update

    torch.save(net.state_dict(), file_prefix + '_recon_net.dat')

#    temp = torch.load(file_prefix + '_recon_net.dat')
#    net.load_state_dict(temp)

    net.zero_grad()

    optimizer2 = optim.Adam(net.parameters(), lr=args.kl_lr)
#    optimizer2 = optim.SGD(net.parameters(), lr=args.kl_lr, momentum = args.kl_mom)

    for epoch in range(args.kl_epochs):
        optimizer2.zero_grad()   # zero the gradient buffers
        ind = torch.randperm(batch_size)[0:args.minibatch_size]
        minibatch = x_mat[ind]
        output = net(minibatch, adjacency, node_feat, edge_feat) 
        loss = kl_loss_forward(minibatch, output, mod_obj, args.num_samples_mc)
        avg_loss = loss.detach().sum()/args.minibatch_size
        loss.backward()
        optimizer2.step()    # Does the update

        to_print =  [epoch, round(avg_loss.item(), 3), round(time.time() - start1, 1)]
        print "Epoch: ", to_print[0], "       loss (kl) = ", to_print[1]

        f.write(' '.join([str(x) for x in to_print]) + '\n')

    f.close()
    torch.save(net.state_dict(), file_prefix + '_net.dat')
    output = net(x_mat, adjacency, node_feat, edge_feat) 
    q_opt = optimal_proposal(mod_obj, x_mat)
    l2_dist_final = reconstruction_loss(output, q_opt)/args.minibatch_size
    print q_opt, output
    print "l2 initial = ", l2_dist_initial.item(), "l2 final = ", l2_dist_final.item()

if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training network using estimated forward KL-based loss for modular functions')
    parser.add_argument('torch_seed', nargs = '?', help='Random seed for torch', type=int, default = 123)
    parser.add_argument('recon_lr', nargs = '?', help='Learning rate for reconstruction phase', type=float, default = 1e-3)
    parser.add_argument('kl_lr', nargs = '?', help='Learning rate for KL-based loss minimisation', type=float, default = 1e-3)
    parser.add_argument('recon_mom', nargs = '?', help='Momentum for reconstruction phase', type=float, default = 0.9)
    parser.add_argument('kl_mom', nargs = '?', help='Momentum for KL-based loss phase', type=float, default = 0.9)
    parser.add_argument('recon_epochs', nargs = '?', help='Number of epochs for reconstruction phase', type=int, default = 0)
    parser.add_argument('kl_epochs', nargs = '?', help='Number of epochs for kl-loss phase', type=int, default = 800)

    parser.add_argument('batch_size', nargs = '?', help='Batch size', type=int, default = 1)
    parser.add_argument('minibatch_size', nargs = '?', help='Minibatch size', type=int, default = 1)
    parser.add_argument('num_samples_mc', nargs = '?', help='#samples to use for loss estimation', type=int, default = 1000)
    args = parser.parse_args()
    args.N = 30 
    args.k = 5 
    torch.manual_seed(args.torch_seed)

    mod_obj = torch.zeros(2, args.N) 
    mod_obj[0] = torch.rand(args.N)*0.5 + 1
    mod_obj[1] = torch.rand(args.N) *0.5 + 0.2 

    x_mat = torch.rand(args.batch_size, args.N)
#    q_opt = optimal_proposal(mod_obj, x_mat)
#    print "exact kl = ", kl_loss_exact_forward(x_mat, q_opt, mod_obj)
#    sys.exit()
#    print x_mat[0], q_opt[0]
#    sys.exit()
    training(x_mat, mod_obj, args)
