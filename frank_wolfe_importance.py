import sys
import numpy as np
import math
import torch
from dpp_objective import getDet as submodObj
from dpp_objective import DPP 
from torch.autograd import Variable
from read_files import read_dpp
#from __future__ import print_function
import logger
from builtins import range
import time

def getProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.prod(temp, 1)

def getLogProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    a = torch.log(pVec)
    b = torch.log(1 - pVec)
    logp = a * sample + b * (1 - sample)
    return logp.sum(1)

def herd_points(probs, num):
    """ Based on Welling & Chen (2010), eqn (18) and (19) """
    x = probs > 0.5
    w = probs - x[-1].float()
    x = x.unsqueeze(0)

    for i in range(num - 1):
        x_next = (w > 0.5)
        w = w + probs - x_next.float() # np.mean(x, 0) - x_next 
        x = torch.cat((x, x_next.unsqueeze(0)))

    return x.float()

def getImportanceWeights(samples_list, nominal, proposal):
    logp_nom = getLogProb(samples_list, nominal)
    logp_prp = getLogProb(samples_list, proposal)
    return torch.exp(logp_nom - logp_prp)

def getImportanceRelax(x_good, x, nsamples, dpp_obj, herd, a): 

    current_sum = Variable(torch.FloatTensor([0]), requires_grad = False) 

    x_prp = (1 - a)*x + a*x_good

    if herd == 1:
        samples_list = herd_points(x_prp, nsamples) 
    else:
        samples_list = Variable(torch.bernoulli(x_prp.repeat(nsamples, 1)))

    w = getImportanceWeights(samples_list, x, x_prp)

    for i in range(nsamples):
#        current_sum = current_sum + (w[i]/w.sum())*dpp(samples_list[i])
        current_sum = current_sum + w[i]*dpp_obj(samples_list[i])

    return current_sum/nsamples

def getCondGrad(grad, k):

    #conditional gradient for cardinality constraints

    N = grad.shape[0]
    top_k = Variable(torch.zeros(N)) #conditional grad
    sorted_ind = torch.sort(grad, descending = True)[1][0:k]
    neg_ind = grad < 0
    top_k[sorted_ind] = 1
    top_k[neg_ind] = 0
    return top_k


def getImportanceGrad(x_good, x, nsamples, dpp_obj, herd, a):

    #Returns the gradient vector of the multilinear relaxation at x as given in Chekuri's paper
    #(See Theorem 1 in nips2012 paper)

    grad = Variable(torch.zeros(dpp_obj.N))

    x_prp = (1 - a)*x + a*x_good

    if herd == 1: 
        samples_list = herd_points(x_prp, nsamples) 
    else:
        samples_list = Variable(torch.bernoulli(x_prp.repeat(nsamples, 1)))

    w = getImportanceWeights(samples_list, x, x_prp)

    for t in range(nsamples):
        sample = samples_list[t] 
        m = torch.zeros(sample.size()) 
        for p in np.arange(dpp_obj.N):
            m[p] = 1
            a = torch.Tensor(np.logical_or(sample.numpy(), m.numpy()).astype(int))
            b = torch.Tensor(np.logical_and(sample.numpy(), np.logical_not(m.numpy())).astype(int))
            grad[p] = grad[p] + w[t]*(dpp_obj(a) - dpp_obj(b))
            m[p] = 0

    return grad/nsamples

def prune(dpp_obj, I):
    #We go through items in I in the order 1...N, and keep items with +ve marginal gain

    items_I = torch.LongTensor([x for x in range(len(I)) if I[x] == 1] )
    sorted_I_items = torch.sort(items_I)[0] #asceding order
    current_set = torch.Tensor([0]*dpp_obj.N)
    for item in sorted_I_items:
        include_sample = current_set.clone()
        include_sample[item] = 1
        marginal_gain = dpp_obj(include_sample) - dpp_obj(current_set) 
        if marginal_gain > 0:
            current_set[item] = 1
#            print "Including item ", item, "    gain = ", marginal_gain
#        else:
#            print "Not including item ", item, "    gain = ", marginal_gain
    temp = [x for x in range(1, dpp_obj.N) if current_set[x] == 1]
    print "Items in pruned set = ", len(temp)
    return current_set

def runImportanceFrankWolfe(dpp_obj, nsamples, k, log_file, opt_file, iterates_file, num_fw_iter, if_herd, x_good, a):

    N = dpp_obj.N 

#    x = Variable(torch.Tensor([1.0*k/N]*N))
    x = Variable(torch.Tensor([1e-3]*N))

    bufsize = 0

    f = open(log_file, 'w', bufsize)
    f2 = open(iterates_file, 'w', bufsize)

    tic = time.clock()

    iter_num = 0
    obj = getImportanceRelax(x_good, x, nsamples, dpp_obj, if_herd, a)
    toc = time.clock()

    print "Iteration: ", iter_num, "    obj = ", obj.item(), "  time = ", (toc - tic),  "   Total/New/Cache: ", dpp_obj.itr_total , dpp_obj.itr_new , dpp_obj.itr_cache

    f.write(str(toc - tic) + " " + str(obj.item()) + " " + str(dpp_obj.itr_total) + '/' + str(dpp_obj.itr_new) + '/' + str(dpp_obj.itr_cache) + "\n") 

    for x_t in x:
        f2.write(str(x_t.item()) + '\n')
    f2.write('\n')

    for iter_num in np.arange(1, num_fw_iter):

        dpp_obj.counter_reset()

        grad = getImportanceGrad(x_good, x,nsamples, dpp_obj, if_herd, a)

        x_star = getCondGrad(grad, k)

        step = 2.0/(iter_num + 2) 

        x = step*x_star + (1 - step)*x

        obj = getImportanceRelax(x_good, x, nsamples, dpp_obj, if_herd, a)
        
        toc = time.clock()

        print "Iteration: ", iter_num, "    obj = ", obj.item(), "  time = ", (toc - tic),  "   Total/New/Cache: ", dpp_obj.itr_total , dpp_obj.itr_new , dpp_obj.itr_cache

        f.write(str(toc - tic) + " " + str(obj.item()) + " " + str(dpp_obj.itr_total) + '/' + str(dpp_obj.itr_new) + '/' + str(dpp_obj.itr_cache) + "\n") 

        for x_t in x:
            f2.write(str(x_t.item()) + '\n')
        f2.write('\n')

    f.close()
    f2.close()

    x_opt = x

    #Round the optimum solution and get function values
    #Follow the contention rounding scheme for knapsack from section 4.5 of the Chekuri et al (2014) paper

    I = Variable(torch.zeros(N)) #independent set (satisfying the cardinality constraint here)
    sorted_ind = torch.sort(x_opt, descending = True)[1][0:k]
    I[sorted_ind] = 1

    #Since DPPs are non-monotone, we need to prune the independent set
    S = prune(dpp_obj, I)

    opt_submod_val = dpp_obj(S) 

    print "Items selected: " + ' '.join([str(x) for x in range(N) if S[x] == 1])
    print "Rounded discrete solution with pruning= ", opt_submod_val.item()
    print "(Rounded discrete solution without pruning = ", dpp_obj(I).item()

    #Save optimum solution and value
    f = open(opt_file, 'w')

    f.write(str(opt_submod_val.item()) + '\n')

    for x_t in x_opt:
        f.write(str(x_t.item()) + '\n')
    f.close()

    return x_opt

if __name__ == '__main__':
    x = torch.Tensor([0.2]*100) 
    I = torch.bernoulli(x)
    N = 100 
    L = read_dpp("/home/pankaj/Sampling/data/input/dpp/data/" + sys.argv[1], N, '/dpp_0')
    dpp_obj = DPP(L)
    prune(dpp_obj, I)
