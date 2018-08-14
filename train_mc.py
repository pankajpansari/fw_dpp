#!/usr/bin/env python

import sys
sys.path.insert(0, '../helpers')
sys.path.insert(1, '../')
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import torch
from torch.autograd import Variable
from graphnet import MyNet
from helpers import getProb, getLogProb, enumerate_all
from dpp_objective import submodObj, constObj, obj1
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import itertools
import argparse
from train_exact import kl_loss_exact

def reconstruction_loss(input, proposal):
    #Reconstruction loss - L2 difference between input and proposal 
    batch_size = input.size()[0]
    temp = input - proposal
    l2_norms = torch.norm(temp, 2, 1)
    return ((l2_norms**2).sum())/batch_size


def kl_loss_mc_uniform(input, proposal, L, nsamples):
    #Estimate the objective function using sets from uniform distribution
    batch_size = input.size()[0]
    obj = Variable(torch.FloatTensor([0]*batch_size)) 
    N = int(L.shape[0])

    uniformP = Variable(torch.FloatTensor([1.0/math.pow(2, N)]))
    
    for t in range(nsamples):
        #draw a sample/set from the uniform distribution
        sample = Variable(torch.bernoulli(torch.FloatTensor([0.5]*N)))
        val = torch.abs(submodObj(L, sample))
#        val = torch.abs(constObj(L, sample))
#        val = torch.abs(obj1(L, sample))
        inputlogP = getLogProb(sample, input)
        proplogP = getLogProb(sample, proposal)
        propP = getProb(sample, proposal)
        inputP = getProb(sample, input)
        obj = torch.add(obj, (propP/uniformP) *(proplogP - (inputlogP + torch.log(val))))
    return obj.mean()/nsamples

def kl_loss_mc_input(input, proposal, L, nsamples):
    #Sampling from proposal distribution
    batch_size = input.size()[0]
    obj = Variable(torch.FloatTensor([0]*batch_size)) 
    N = int(L.shape[0])

    for t in range(batch_size):
        proposal_t = proposal[t, :].unsqueeze(0)
        input_t = input[t, :].unsqueeze(0)
        obj_t = Variable(torch.FloatTensor([0])) 
        assert(proposal_t.nonzero().nelement() > 0)
        assert(input_t.nonzero().nelement() > 0)
        for p in range(nsamples):
            #draw a sample/set from the uniform distribution
            sample = Variable(torch.bernoulli(input_t.squeeze().data))
            val = torch.abs(submodObj(L, sample))
            inputP = getProb(sample, input_t)
            propP = getProb(sample, proposal_t)
#            print inputP.item(), propP.item()
#            assert(inputP > 1e-5)
#            assert(propP > 1e-5)
#            print (propP/inputP).item(), torch.log(propP).item(), torch.log(torch.abs(val)*inputP).item()
            obj_t += (propP/inputP)*(torch.log(propP) - torch.log(torch.abs(val)*inputP))
            if math.isinf(obj_t.item()):
                print sample, input_t, proposal_t
                print propP.item(), torch.abs(val).item(), inputP.item()
                print torch.abs(val)*inputP
                print ((propP/inputP)*(torch.log(propP) - torch.log(torch.abs(val)*inputP))).item()
            assert(math.isinf(obj_t.item()) == False)

            if math.isnan(obj_t.item()):
                print sample, input_t, proposal_t
                print propP.item(), torch.abs(val).item(), inputP.item()
            assert(math.isnan(obj_t.item()) == False)
        obj[t] = obj_t/nsamples
    return obj.mean()

def kl_loss_mc_uniform_multipleL(input, proposal, L_mat, nsamples):
    #Estimate the objective function using sets from uniform distribution
    batch_size = input.size()[0]
    obj = Variable(torch.FloatTensor([0]*batch_size)) 
    N = int(np.sqrt(int(L_mat[0].shape[0])))

    uniformP = Variable(torch.FloatTensor([1.0/math.pow(2, N)]))
    for t in range(nsamples):
        #draw a sample/set from the uniform distribution
        sample = Variable(torch.bernoulli(torch.FloatTensor([0.5]*N)))
        val = torch.abs(submodObj_multipleL(L_mat, sample))
        inputP = getProb(sample, input)
        y = inputP * val
        propP = getProb(sample, proposal)
        obj += (propP*torch.log(propP) - propP*torch.log(y))/uniformP
    return obj.sum()/(nsamples*batch_size)

def training_mc(input_x, phi, L, adjacency, node_feat, edge_feat, N, net, lr1, lr2, mom, minibatch_size, num_samples_mc, file_prefix):

    optimizer = optim.Adam(net.parameters(), lr=lr1)
#    optimizer = optim.SGD(net.parameters(), lr=lr1)

    reconstruction_list = []
    kl_list = []
    var_list = []
    epoch_list = []

    f = open(file_prefix + '_training_log.txt', 'a')
    batch_size = int(input_x.shape[0]) 


    for epoch in range(100):
        #get minibatch
        optimizer.zero_grad()   # zero the gradient buffers
        ind = torch.randperm(batch_size)[0:minibatch_size]
        minibatch = input_x[ind]
        output = net(minibatch, phi, adjacency, node_feat, edge_feat) 
        loss = reconstruction_loss(minibatch , output)
        print "Epoch: ", epoch, "       loss (l2 reconstruction) = ", loss.data.numpy()
        f.write("Epoch: " +  str(epoch) +  "       loss (l2 reconstruction) = " + str(loss.data.item()) + "\n")
        loss.backward()
        optimizer.step()    # Does the update

        reconstruction_list.append(loss.item())
        if epoch % 10 == 0:
            plt.plot(reconstruction_list)
            plt.xlabel('Number of epochs')
            plt.ylabel('L2 reconstruction loss')
            plt.savefig(file_prefix + '_recon.png', bbox_inches='tight')
            plt.gcf().clear()


#    torch.save(net.state_dict(), file_prefix + '_net.dat')
#    print input_x 
#    print net(input_x, phi, adjacency, node_feat, edge_feat) 
#
#    sys.exit()

#    net.load_state_dict(torch.load(file_prefix + '_net.dat'))
    
    optimizer2 = optim.SGD(net.parameters(), lr=lr2, momentum = mom)

    output =  net(input_x, phi, adjacency, node_feat, edge_feat) 
#    for t in range(input_x.size()[0]):
#        print input_x[t], output[t] 

    for epoch in range(100):
        optimizer2.zero_grad()   # zero the gradient buffers
        ind = torch.randperm(batch_size)[0:minibatch_size]
        minibatch = input_x[ind]
        output = net(minibatch, phi, adjacency, node_feat, edge_feat) 
#        loss = kl_loss_mc_uniform(minibatch, output, L, num_samples_mc)
        loss = kl_loss_mc_input(minibatch, output, L, num_samples_mc)
#        loss = kl_loss_exact(minibatch, output, L)
        print "Epoch: ", epoch, "       loss = ", loss.item()
        f.write("Epoch: " +  str(epoch) +  "       loss = " + str(loss.item()) + "\n")
        loss.backward()
        optimizer2.step()    # Does the update

        kl_list.append(loss.item())
        if epoch % 10 == 0:
            plt.plot(kl_list)
            plt.xlabel('Number of epochs')
            plt.ylabel('Loss')
            plt.savefig(file_prefix + '_loss.png', bbox_inches='tight')
            plt.gcf().clear()

#    proposal = net(input[:, 0:N])
#    print "input = ", input[:, 0:N].data
#    print "proposal = ", proposal.data
    torch.save(net.state_dict(), file_prefix + '_net.dat')
    f.close()

def main():

    parser = argparse.ArgumentParser(description='Training network using estimated KL-based loss')
    parser.add_argument('torch_seed', nargs = '?', help='Random seed for torch', type=int, default = 123)
    parser.add_argument('N',metavar = 'dpp_size', nargs = '?', help='# of items in DPP', type=int, default = 20)
    parser.add_argument('architecture_choice', nargs = '?', help='Choice of network architecture', type=int, default = 4)
    parser.add_argument('lr1', metavar = 'lr_recon', nargs = '?', help='Learning rate for reconstruction phase', type=float, default = 1e-2)
    parser.add_argument('lr2', metavar = 'lr_kl', nargs = '?', help='Learning rate for KL-based loss minimisation', type=float, default = 1e-5)
    parser.add_argument('mom', nargs = '?', help='Momentum', type=float, default = 1e-3)
    parser.add_argument('num_DPP', nargs = '?', help='# of DPPs', type=int, default = 1)
    parser.add_argument('batch_size', nargs = '?', help='Batch size', type=int, default = 100)
    parser.add_argument('minibatch_size', nargs = '?', help='Minibatch size', type=int, default = 100)
    parser.add_argument('num_samples_mc', nargs = '?', help='#samples to use for loss estimation', type=int, default = 100)
    args = parser.parse_args()

    start = time.time()
    np_seed = 456 

    wdir = '/home/pankaj/Sampling/data/working/15_05_2018/'
    file_prefix = wdir + '/dpp_' + str(args.torch_seed) + '_' + str(np_seed) + '_' + str(args.N) + '_' + str(args.architecture_choice) + '_' + str(args.lr1) + '_' + str(args.lr2) + '_' + str(args.mom) + '_' + str(args.num_DPP) +  '_' + str(args.batch_size) + '_' + str(args.minibatch_size) + '_' + str(args.num_samples_mc)

    torch.manual_seed(args.torch_seed)
    np.random.seed(np_seed)

    w_m = 10
    scale = 1
    num_feat = 10 

    #feature vectors as columns
    phi = Variable(torch.randn(num_feat, args.N))
    #normalise the features to have 2-norm = 1
    phi = torch.nn.functional.normalize(phi, p = 2, dim = 0)

    #Generate a fixed vector of norm 1 to obtain values of m for quality
    fixed_v = Variable(torch.randn(num_feat, 1))
    fixed_v = torch.nn.functional.normalize(fixed_v , p = 2, dim = 0)
    m = torch.mm(phi.t(), fixed_v)
    quality = torch.exp(0.5*w_m*m)

    N = args.N
    #Gram matrix
    S = torch.mm(phi.t(), phi)
    M = quality * Variable(torch.eye(N))
    L = scale * (torch.mm(M, S).mm(M))

    net = MyNet()
    #generate input data uniformly of size batch_size for each dpp
    input_x = Variable(torch.rand(args.batch_size, args.N), requires_grad = True) 

#    #Concatenate x and flattened L 
#    for p in range(L_mat.shape[0]):
#        temp = L_mat[p].repeat(args.batch_size, 1)
#        temp2 = torch.cat((input_x, temp), dim = 1)
#        if p == 0:
#            input = temp2.clone()
#        else:
#            input = torch.cat((input, temp2), dim = 0)

    #Do a forward pass
#    L = input[0, N:].view(N, N)

    adjacency = Variable(torch.ones(N, N))
    for t in range(N):
        adjacency[t, t] = 0
    adjacency = adjacency.unsqueeze(0)
    adjacency = adjacency.repeat(args.batch_size, 1, 1)

    #node features
    scale_v = Variable(torch.FloatTensor([scale]*args.N)).unsqueeze(1)
    wm_v = Variable(torch.FloatTensor([w_m]*args.N)).unsqueeze(1)
    bias = Variable(torch.FloatTensor([1.0]*args.N)).unsqueeze(1)
    single_sample_node_feat = torch.cat([scale_v, wm_v, quality, m, phi.t(), fixed_v.repeat(1, args.N).t(), bias], dim = 1)
    batch_node_feat = single_sample_node_feat.repeat(args.batch_size, 1, 1)

    #edge features
    scale_M = Variable(torch.FloatTensor(1, args.N, args.N).fill_(scale))
    wm_M = Variable(torch.FloatTensor(1, args.N, args.N).fill_(w_m))
    bias_M = Variable(torch.FloatTensor(1, args.N, args.N).fill_(1))
    quality_1 = quality.repeat(1, 1, args.N)
    quality_2 = quality.t().repeat(1, args.N, 1)
    m_1 =  m.repeat(1, 1, args.N)
    m_2 =  m.t().repeat(1, args.N, 1)

    similarity = torch.mm(phi.t(), phi).unsqueeze(0)

    feature_cat_M = torch.empty(3*num_feat, args.N, args.N)
    for i in range(args.N):
        for j in range(args.N):
            feature_cat_M[:, i, j] = torch.cat((phi[:, i], phi[:, j], fixed_v.squeeze()))
    single_sample_edge_feat = torch.cat((scale_M, wm_M, bias_M, similarity, quality_1, quality_2, m_1, m_2, feature_cat_M), dim = 0) 
    batch_edge_feat =  single_sample_edge_feat.repeat(args.batch_size, 1, 1, 1)

    print "Size of training data = " + str(args.batch_size*args.num_DPP)
    f = open(file_prefix + '_training_log.txt', 'a')
    f.write("Size of training data = " + str(args.batch_size*args.num_DPP) + "\n")


    print "Training using estimated KL-based loss"
    start1 = time.time()
    training_mc(input_x, phi, L, adjacency, batch_node_feat, batch_edge_feat, args.N, net, args.lr1, args.lr2, args.mom, args.minibatch_size, args.num_samples_mc, file_prefix)
    end1 = time.time()
    print "Training completed in time: ", end1 - start1, "s"
    f.write("Training time = " + str(end1 - start1) + "\n")
#    print "Loading network"
#    net.load_state_dict(torch.load(file_prefix + '_net.dat'))

if __name__ == '__main__':
    main()
