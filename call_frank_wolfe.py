import sys
import numpy as np
import math
import torch
from influence import ic_model as submodObj
from torch.autograd import Variable
from frank_wolfe import runFrankWolfe
from frank_wolfe_importance import runImportanceFrankWolfe
from read_files import read_dpp
#from variance import convex_var
import time
import subprocess
import argparse
np.random.seed(1234)

def call_FrankWolfe(N, dpp_id, k, nsamples_mlr, num_fw_iter,  if_herd, if_sfo_gt, a, torch_seed):

    dpp_file = "/home/pankaj/Sampling/data/input/dpp/data/dpp_100_0.5_0.5_200_0_0.1_5.h5" 

    L = read_dpp(dpp_file, N, '/dpp_' + str(dpp_id)) 

    dirw = './workspace'

    temp = dirw + '/fw_log_N_' + str(N) + '_' + str(dpp_id) 

    log_file = '_'.join(str(x) for x in [temp, k, nsamples_mlr, num_fw_iter, if_herd, if_sfo_gt, a, torch_seed]) + '.txt'

    temp = dirw + '/fw_opt_N_' + str(N) + '_' + str(dpp_id) 

    opt_file = '_'.join(str(x) for x in [temp, k, nsamples_mlr, num_fw_iter, if_herd, if_sfo_gt, a, torch_seed]) + '.txt'

    temp = dirw + '/iterates_N_' + str(N) + '_' + str(dpp_id) 

    iterates_file = '_'.join(str(x) for x in [temp, k, nsamples_mlr, num_fw_iter, if_herd, if_sfo_gt, 0, torch_seed]) + '.txt'

    x_opt = runImportanceFrankWolfe(L, nsamples_mlr, k, log_file, opt_file, iterates_file, num_fw_iter, if_herd, x_good, a)

def main():

    tic = time.clock()

    parser = argparse.ArgumentParser(description='Running Frank-Wolfe on given DPP')
    parser.add_argument('N', help='Number of items in DPP', type=int)
    parser.add_argument('dpp_id', help='Id of dpp file', type=int)
    parser.add_argument('k', help='Cardinality constraint', type=int)
    parser.add_argument('nsamples_mlr', help='Number of samples for multilinear relaxation estimation', type=int)
    parser.add_argument('num_fw_iter', help='Number of iterations of Frank-Wolfe', type=int)
    parser.add_argument('if_herd', help='True if herding', type=int)
    parser.add_argument('if_sfo_gt', help='True if greedy ground-truth to be used during importance sampling', type=int)
    parser.add_argument('a', help='Convex combination coefficient', type=float)
    parser.add_argument('torch_seed', help='Torch random seed', type=int)

    args = parser.parse_args()
    
    N = args.N
    dpp_id = args.dpp_id
    k = args.k #cardinality constraint
    nsamples_mlr = args.nsamples_mlr #draw these many sets from x for multilinear relaxation
    num_fw_iter = args.num_fw_iter 
    if_herd = args.if_herd
    if_sfo_gt = args.if_sfo_gt
    a = args.a
    torch_seed = args.torch_seed

    torch.manual_seed(torch_seed) 

    call_FrankWolfe(N, dpp_id, k, nsamples_mlr, num_fw_iter, if_herd, if_sfo_gt, a, torch_seed)

    print "Compeleted in " + str(time.clock() - tic) + 's'

if __name__ == '__main__':
    main()
