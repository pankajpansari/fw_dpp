#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import visdom
import torch

# Gather our code in a main() function
def main():

    vis = visdom.Visdom()

    b = np.loadtxt('log_p_0.1.txt')
    x = torch.from_numpy(b)
    niter_list = [10, 100, 1000]
    N_list = [6, 7, 8]
    ngraphs = 4 
    nsamples = 5 

    for N in N_list:
        m = []
        for niter in niter_list:
            t = []
            for i in range(ngraphs):
                for j in range(nsamples):
                    y = x[(x[:, 0] == N) & (x[:, 1] == niter) & (x[:, 2] == i) & (x[:, 3] == j)]
                    t.append(y[:, 5].var())
            m.append(sum(t)/float(len(t)))
        plt.plot(niter_list, m, label = str(int(math.pow(2, N))) + ' nodes')
    plt.legend()
    plt.xlabel('Number of Monte-carlo simulations for influence propagation')
    plt.ylabel('Variance in influence values')
    plt.title('Influence propagation propability = 0.1')
    vis.matplot(plt)
    
    plt.clf()
    x = np.loadtxt('log_p_0.01.txt')
    niter_list = [10, 100, 1000]
    N_list = [6, 7, 8, 9]

    for N in N_list:
        t = []
        for niter in niter_list:
            y = x[(x[:, 0] == N) & (x[:, 1] == niter) & (x[:, 2] == 0) & (x[:, 3] == 0)]
            t.append(y[:, 5].var())
        plt.plot(niter_list, t, label = str(int(math.pow(2, N))) + ' nodes')
    plt.legend()
    plt.xlabel('Number of Monte-carlo simulations for influence propagation')
    plt.ylabel('Variance in influence values')
    plt.title('Influence propagation propability = 0.01')
#    vis.matplot(plt)
    vis.text('100 iterations give sufficiently low variance')

def plot_ic_model_variance():

    b = np.loadtxt('/home/pankaj/Sampling/data/working/17_07_2018/merge_p_0.2.txt')
    num_mc_sim_list = [10, 100, 1000, 10000]
    var_list = []
    time_list = [] 
    for num_mc_sim in num_mc_sim_list:
        var_val = b[b[:, 0] == num_mc_sim, 3]
        exec_time = b[b[:, 0] == num_mc_sim, 4]
        var_list.append(np.mean(var_val))
        time_list.append(np.mean(exec_time))
    
    print num_mc_sim_list
    print var_list, time_list

def get_relaxation_variance():

    b = np.loadtxt('/home/pankaj/Sampling/data/working/23_07_2018/merge_relax_10.txt')
    var_list = []
    time_list = [] 
    print np.var(b[:, 1]), np.mean(b[:, 2]) 

if __name__ == '__main__':
    get_relaxation_variance()

