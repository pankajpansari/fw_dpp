from read_files import read_dpp
from dpp_objective import DPP 
import sys
import numpy as np
import math
import torch
from torch.autograd import Variable
import logger
from builtins import range
import time
np.random.seed(1234)
torch.manual_seed(1234) 

def print_list(text_list, val_list):
    temp = list(zip(text_list, val_list))
    for (a, b) in temp:
        print a, ': ', str(round(b, 2)), '    ',
    print

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

def getRelax(dpp, x, nsamples, herd = 0): 

    current_sum = Variable(torch.FloatTensor([0]), requires_grad = False) 

    if herd == 1:
        samples_list = herd_points(x, nsamples) 
    else:
        samples_list = Variable(torch.bernoulli(x.repeat(nsamples, 1)))

    for sample in samples_list:
        val = dpp(sample)
        current_sum = current_sum +  val

    return current_sum/nsamples

def getGrad(dpp, x, nsamples, herd = 0):

    #Returns the gradient vector of the multilinear relaxation at x as given in Chekuri's paper
    #(See Theorem 1 in nips2012 paper)

    N = dpp.N 
    grad = Variable(torch.zeros(N))

    if herd == 1: 
        samples_list = herd_points(x, nsamples) 
    else:
        samples_list = Variable(torch.bernoulli(x.repeat(nsamples, 1)))

    for t in range(nsamples):
        sample = samples_list[t] 
        m = torch.zeros(sample.size()) 
        for p in np.arange(N):
            m[p] = 1
            a = torch.Tensor(np.logical_or(sample.numpy(), m.numpy()).astype(int))
            b = torch.Tensor(np.logical_and(sample.numpy(), np.logical_not(m.numpy())).astype(int))
            grad[p] = grad[p] + (dpp(a) - dpp(b))
            m[p] = 0
    return grad*1.0/nsamples

def getCondGrad(grad, k):

    #conditional gradient for cardinality constraints

    N = grad.shape[0]
    top_k = Variable(torch.zeros(N)) #conditional grad
    sorted_ind = torch.sort(grad, descending = True)[1][0:k]
    neg_ind = grad < 0
    top_k[sorted_ind] = 1
    top_k[neg_ind] = 0
    return top_k


def rounding(x, k):
    rounded_x = torch.zeros(x.shape) #conditional grad
    sorted_ind = torch.sort(x, descending = True)[1][0:k]
    rounded_x[sorted_ind] = 1
    return rounded_x

def prune(dpp, I):
    #We go through items in I in the order 1...N, and keep items with +ve marginal gain

    items_I = torch.LongTensor([x for x in range(len(I)) if I[x] == 1] )
    sorted_I_items = torch.sort(items_I)[0] #asceding order
    current_set = torch.Tensor([0]*dpp.N)
    for item in sorted_I_items:
        include_sample = current_set.clone()
        include_sample[item] = 1
        marginal_gain = dpp(include_sample) - dpp(current_set) 
        if marginal_gain > 0:
            current_set[item] = 1
    temp = [x for x in range(dpp.N) if current_set[x] == 1]
    print "Items in pruned set = ", len(temp)
    return current_set


def runFrankWolfe(dpp, args, log_file, opt_file, iterates_file, if_herd = 0):

    x = torch.Tensor([1.0*args.k/args.N]*args.N)

    bufsize = 0

    f = open(log_file, 'w', bufsize)
    f2 = open(iterates_file, 'w', bufsize)

    tic = time.clock()

    mlr = getRelax(dpp, x, args.num_samples_mc)
    toc = time.clock()

    rounded_val = []
    rounded_val_best = []
    rounded_best = -10

    text_list = ['Iteration', 'mlr', 'time']
    val_list = [0, mlr.item(), (toc - tic)]
    print_list(text_list, val_list)

    f.write(str(toc - tic) + " " + str(mlr.item()) + " " + str(dpp.itr_total) + '/' + str(dpp.itr_new) + '/' + str(dpp.itr_cache) + "\n") 

    for x_t in x:
        f2.write(str(x_t.item()) + '\n')
    f2.write('\n')

    for iter_num in np.arange(1, args.num_fw_iter):

        dpp.counter_reset()

        grad = getGrad(dpp, x, args.num_samples_mc, )

        x_star = getCondGrad(grad, args.k)

        step = 2.0/(iter_num + 2) 

        x = step*x_star + (1 - step)*x

        mlr = getRelax(dpp, x, args.num_samples_mc)
        
        toc = time.clock()

        val_list = [iter_num, mlr.item(), (toc - tic)]
        print_list(text_list, val_list)
#        print "Iteration: ", iter_num, "    mlr = ", mlr.item(), "  time = ", (toc - tic),  "   Total/New/Cache: ", dpp.itr_total , dpp.itr_new , dpp.itr_cache

        f.write(str(toc - tic) + " " + str(mlr.item()) + " " + str(dpp.itr_total) + '/' + str(dpp.itr_new) + '/' + str(dpp.itr_cache) + "\n") 

        for x_t in x:
            f2.write(str(x_t.item()) + '\n')
        f2.write('\n')

    f.close()

    #Round the optimum solution and get function values
    rounded_x = rounding(x, args.k)

    #Since DPPs are non-monotone, we need to prune the independent set
    pruned_x = prune(dpp, rounded_x)

    print pruned_x

    opt_val = dpp(pruned_x) 

    print "Final rounded value = ", opt_val.item()

    #Save optimum value, fractional solution, and rounded+pruned solution 
    f = open(opt_file, 'w')

    f.write(str(opt_val.item()) + '\n')

    for t in x:
        f.write(str(t.item()) + '\n')

    for t in pruned_x:
        f.write(str(t.item()) + '\n')

    f.close()

    return x

def main():
    grad = torch.randn(10)

if __name__ == '__main__':
    main()
