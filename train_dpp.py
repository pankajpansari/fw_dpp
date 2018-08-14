#!/usr/bin/env python

import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import torch
from torch.autograd import Variable
from variance import plot_variance
from variance import multilinear_importance 
from helpers import getProb
from helpers import getLogProb
from influence import objective 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import itertools

ifDebug = 0

def kl_loss_mc_uniform(input, proposal, g, nsamples):
    #Estimate the objective function using sets from uniform distribution
    batch_size = input.size()[0]
    obj = Variable(torch.FloatTensor([0]*batch_size)) 
    N = int(L.shape[0])

    uniformP = Variable(torch.FloatTensor([1.0/math.pow(2, N)]))
    
#    print input, proposal
    for t in range(nsamples):
        #draw a sample/set from the uniform distribution
        sample = Variable(torch.bernoulli(torch.FloatTensor([0.5]*N)))
        val = objective(g) 
#        val = torch.abs(constObj(L, sample))
#        val = torch.abs(obj1(L, sample))
        inputlogP = getLogProb(sample, input)
        proplogP = getLogProb(sample, proposal)
        propP = getProb(sample, proposal)
        inputP = getProb(sample, input)
        obj = torch.add(obj, (propP/uniformP) *(proplogP - (inputlogP + torch.log(val))))
    return obj.mean()/nsamples

def kl_loss_mc_proposal(input, proposal, L, nsamples):
    #Sampling from proposal distribution
    batch_size = input.size()[0]
    obj = Variable(torch.FloatTensor([0]*batch_size)) 
    N = int(L.shape[0])

    for t in range(batch_size):
        proposal_t = proposal[t, :].unsqueeze(0)
        input_t = input[t, :].unsqueeze(0)
        obj_t = Variable(torch.FloatTensor([0])) 
        for p in range(nsamples):
            #draw a sample/set from the uniform distribution
            sample = Variable(torch.bernoulli(proposal_t.squeeze().data))
            val = torch.abs(submodObj(L, sample))
            input_tP = getProb(sample, input_t)
            y = input_tP * val
            z = torch.log(y)
            propP = getProb(sample, proposal_t)
            obj_t += torch.log(propP) - torch.log(y)
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

def reconstruction_loss(input, proposal):
    #Reconstruction loss - L2 difference between input and proposal 
    batch_size = input.size()[0]
    temp = input - proposal
    l2_norms = torch.norm(temp, 2, 1)
    return ((l2_norms**2).sum())/batch_size

def grad_importance(L, x, net, use_net, nsamples):
#Returns the gradient vector of the multilinear relaxation at x as given in Chekuri's paper
#(See Theorem 1 in nips2012 paper)
    N = L.shape[0]
    grad = Variable(torch.zeros(N))
    input_sample = torch.cat((x, L.view(1, N*N)), dim = 1)
    if use_net == 1:
#        z = net(input_sample, 0) 
        z = net(x, 0) 
    else:
        z = x.clone()

#    if use_net == 1:
#        print x, z
    for p in np.arange(N):
        z_include = z.clone()
        z_exclude  = z.clone() 
        z_include[:, p] = 1
        z_exclude[:, p] = 0

        x_include = x.clone()
        x_exclude  = x.clone() 
        x_include[:, p] = 1
        x_exclude[:, p] = 0

        grad[p] =  multilinear_importance(x_include, z_include, L, nsamples) - multilinear_importance(x_exclude, z_exclude, L, nsamples)

    return grad

def getCondGrad(grad):
    n = grad.shape[0]
    y = Variable(torch.zeros(n)) #conditional grad
    for p in range(n):
        if grad.data[p] > 0:
            y[p] = 1
        else:
            y[p] = 0
    return y

def FW_importance(L, nsamples, net, file_prefix, use_net, save_iterates):

    N = L.shape[0]
    x = Variable(torch.Tensor(1, N))
    X = Variable(torch.Tensor(1, (N + 1)*N))
    x[0] = Variable(torch.Tensor([0.5]*N))


    for iter_num in np.arange(1, 50):

        grad = grad_importance(L, x, net, use_net, nsamples)
        x_star = getCondGrad(grad)

        step = 2.0/(iter_num + 2) 

        fenchel_gap = grad.dot(x_star - x)

        x = step*x_star + (1 - step)*x
    
        temp = L.view(1, N*N)

        #note: we are saving the data of each sample as (x, L flattened)
        if save_iterates == 1:
            sample_data = torch.cat((x, temp), dim = 1)

            if iter_num == 1:
                X[0] = sample_data.clone()
            else:
                X = torch.cat((X, sample_data))
     
#        if fenchel_gap.data[0] < 0.01:
#            break

    if save_iterates == 1:
        f = open(file_prefix + '_training_input.dat', 'ab')
        np.savetxt(f, X.data.numpy())
        f.close()

    currentVal = getExactRelax(L, x)
    return x, currentVal.data[0], iter_num

def entropy_mc(output, nsamples):
    ent_sum = 0
    batch_size = output.size()[0]
    for t in range(nsamples):
        sample = torch.bernoulli(output)
        ent_sum += -torch.log(getProb(sample, output))
    return ent_sum.sum()/(nsamples*batch_size)
                    
def training_mc(input, g, N, net, lr1, lr2, mom, minibatch_size, num_samples_mc, file_prefix):

    optimizer = optim.Adam(net.parameters(), lr=lr1)
#    optimizer = optim.SGD(net.parameters(), lr=lr1)

    reconstruction_list = []
    kl_list = []
    var_list = []
    epoch_list = []

    f = open(file_prefix + '_training_log.txt', 'a')
    batch_size = int(input.shape[0]) 

    for epoch in range(800):
        #get minibatch
        optimizer.zero_grad()   # zero the gradient buffers
        ind = torch.randperm(batch_size)[0:minibatch_size]
        minibatch = input[ind]
        output = net(minibatch[:, 0:N], 1) 
        loss = reconstruction_loss(minibatch[:, 0:N] , output)
        print "Epoch: ", epoch, "       loss (l2 reconstruction) = ", loss.data[0]
        f.write("Epoch: " +  str(epoch) +  "       loss (l2 reconstruction) = " + str(loss.data[0]) + "\n")
        loss.backward()
        optimizer.step()    # Does the update

        reconstruction_list.append(loss.data[0])
        if epoch % 10 == 0:
            plt.plot(reconstruction_list)
            plt.xlabel('Number of epochs')
            plt.ylabel('L2 reconstruction loss')
            plt.savefig(file_prefix + '_recon.png', bbox_inches='tight')
            plt.gcf().clear()


    torch.save(net.state_dict(), file_prefix + '_net.dat')
#    sys.exit()

#    net.load_state_dict(torch.load(file_prefix + '_net.dat'))

    optimizer2 = optim.Adam(net.parameters(), lr=lr2)
#    optimizer2 = optim.SGD(net.parameters(), lr=lr2, momentum = mom)

    for epoch in range(200):
        optimizer2.zero_grad()   # zero the gradient buffers
        ind = torch.randperm(batch_size)[0:minibatch_size]
        minibatch = input[ind]
#        minibatch = input
        output = net(minibatch[:, 0:N]) 
        loss = kl_loss_mc_uniform(minibatch[:, 0:N], output, g, num_samples_mc)
        print "Epoch: ", epoch, "       loss = ", loss.data[0]
        f.write("Epoch: " +  str(epoch) +  "       loss = " + str(loss.data[0]) + "\n")
        loss.backward()

        optimizer2.step()    # Does the update

        kl_list.append(loss.data[0])
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

#    if len(sys.argv) < 11:
#        print "Usage: python mlp_dpp.py torch_seed dpp_size architecture_choice lr_recon lr_kl momentum #DPPs batch_size minibatch_size num_samples_mc if_exact_kl"
#        print "python mlp_dpp.py 123 8 1 0.0001 0.001 0.01 1 100 10 10 1"
        sys.exit()
    start = time.time()
    torch_seed = int(sys.argv[1])
    np_seed = 456 
    N = int(sys.argv[2])
    architecture_choice = int(sys.argv[3])
    lr1 = float(sys.argv[4])
    lr2 = float(sys.argv[5])
    mom = float(sys.argv[6])
    num_graphs = int(sys.argv[7])
    batch_size = int(sys.argv[8])
    minibatch_size = int(sys.argv[9])
    num_samples_mc = int(sys.argv[10])
    if_exact = int(sys.argv[11])
    wdir = str(os.environ['result_dir'])
 #   wdir = '/home/pankaj/Sampling/data/working/03_04_2018/'
    file_prefix = wdir + '/graph_' + str(torch_seed) + '_' + str(np_seed) + '_' + str(N) + '_' + str(architecture_choice) + '_' + str(lr1) + '_' + str(lr2) + '_' + str(mom) + '_' + str(num_DPP) +  '_' + str(batch_size) + '_' + str(minibatch_size) + '_' + str(num_samples_mc) + '_' + str(if_exact)

    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)

    w_m = 0.1
    scale = 0.1
    for t in range(num_DPP):
        phi = Variable(torch.randn(N, N))
        S = torch.mm(phi, torch.transpose(phi, 0, 1))
        m = Variable(torch.randn(N, 1))
        M = torch.sqrt(torch.exp(w_m * m))
        temp = torch.sqrt(torch.exp(w_m * m)) 
        M = torch.diag(torch.squeeze(temp))
        L = scale * (torch.mm(M, S).mm(M))
        L = L.view(1, N * N)
        if t == 0:
            L_mat = L.clone()
        else:
            L_mat = torch.cat((L_mat, L))

#    enumerate_all(L.view(N, N))
#    sys.exit()
    if architecture_choice == 1:
        net = Net(N)
    if architecture_choice == 2:
        net = Net2(N)
    if architecture_choice == 3:
        net = Net3(N)
    if architecture_choice == 4:
        net = Net4(N)

    #generate input data uniformly of size batch_size for each dpp
#    input_x = Variable(torch.rand(batch_size, N)) 
    input_x = Variable(torch.rand(batch_size, N)) 

    for p in range(L_mat.shape[0]):
        temp = L_mat[p].repeat(batch_size, 1)
        temp2 = torch.cat((input_x, temp), dim = 1)
        if p == 0:
            input = temp2.clone()
        else:
            input = torch.cat((input, temp2), dim = 0)

    print "Size of training data = " + str(batch_size*num_DPP)
    f = open(file_prefix + '_training_log.txt', 'a')
    f.write("Size of training data = " + str(batch_size*num_DPP) + "\n")


    print "Training network"
    start1 = time.time()
    if if_exact == 1:
        print "Exact training"
        training_exact(input, N, net, lr1, lr2, mom, minibatch_size, file_prefix)
    else:
        print "MC training"
        training_mc(input, N, net, lr1, lr2, mom, minibatch_size, num_samples_mc, file_prefix)
    end1 = time.time()
    f.write("Training time = " + str(end1 - start1) + "\n")
#    print "Loading network"
#    net.load_state_dict(torch.load(file_prefix + '_net.dat'))

    print "Computing variances"
    start1 = time.time()
    plot_variance(input, N, net, file_prefix)
    end1 = time.time()
    f.write("Variance computation time = " + str(end1 - start1) + "\n")
    f.write("Full time = " + str(end1 - start) + "\n")

    print "Computing improvement in convergence rates"
    f.write("Computing improvement in convergence rates" + "\n")
    #look at improvement in optimum values
    sample_list = [int(math.pow(2, t)) for t in range(1, N - 2)]
    iter_num_ratio = np.array([])

    for nsample in sample_list:

        print "Using proposals for FW,  #samples = ", nsample
        f.write("Using proposals for FW,  #samples = " + str(nsample) + "\n")

        iter_list_x = np.array([])
        iter_list_y = np.array([])

        for p in range(L_mat.shape[0]):

            print "DPP# = ", p
            f.write("DPP#= " + str(p) + "\n")
            L = L_mat[p].view(N, N)

            for t in range(5): 
                print "Instance = ", t
                dummy1, dummy2, iter_num_x = FW_importance(L, nsample, net, file_prefix, 0, 0)
                iter_list_x = np.append(iter_list_x, int(iter_num_x))
                dummy1, dummy2, iter_num_y = FW_importance(L, nsample, net, file_prefix, 1, 0)
                iter_list_y = np.append(iter_list_y, int(iter_num_y))
                print "Without network: ", iter_num_x, "     with network: ", iter_num_y 
                f.write("Without network: "+ str(iter_num_x) + "     with network: " + str(iter_num_y) + "\n")
                if ifDebug == 1:
                    print iter_list_x, iter_list_y
        iter_num_ratio = np.append(iter_num_ratio, np.mean(iter_list_y/iter_list_x))
        print "ratio = ", np.mean(iter_list_y/iter_list_x)
        f.write("ratio = " + str(np.mean(iter_list_y/iter_list_x)) + "\n")
    plt.plot(sample_list, iter_num_ratio)
    plt.xlabel('Samples used for each FW run')
    plt.ylabel('ratio of iteration numbers for convergence (fenchel gap 0.1) (learned/naive)')
    plt.savefig(file_prefix + '_iter_num_ratio.png', bbox_inches='tight')
    f.close()

if __name__ == '__main__':
    main()

