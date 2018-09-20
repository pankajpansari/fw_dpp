import sys
import numpy as np
import math
import torch
from torch.autograd import Variable
#from frank_wolfe_importance import runImportanceFrankWolfe
from frank_wolfe import runFrankWolfe
from read_files import read_dpp
#from variance import convex_var
import time
import subprocess
import argparse
from dpp_objective import DPP 
np.random.seed(1234)

wdir = './workspace'

def call_FrankWolfe(dpp, args):

    args_list = [args.torch_seed, args.dpp_id, args.N, args.k, args.num_samples_mc, args.num_fw_iter]

    file_prefix = wdir + '/dpp_' + '_'.join([str(x) for x in args_list])

    log_file = file_prefix + '_fw_simple_log.txt'
    opt_file = file_prefix + '_fw_simple_opt.txt'
    iterates_file = file_prefix + '_fw_simple_iterates.txt'

    x_opt = runFrankWolfe(dpp, args, log_file, opt_file, iterates_file)

def main():

    tic = time.clock()

    parser = argparse.ArgumentParser(description='Running Frank-Wolfe on given DPP')
    parser.add_argument('torch_seed', help='Torch random seed', type=int)
    parser.add_argument('dpp_file', help='DPP file', type=str)
    parser.add_argument('dpp_id', help='Id of dpp file', type=int)
    parser.add_argument('N', help='Number of items in DPP', type=int)
    parser.add_argument('k', help='Cardinality constraint', type=int)
    parser.add_argument('num_samples_mc', help='Number of samples for multilinear relaxation estimation', type=int)
    parser.add_argument('num_fw_iter', help='Number of iterations of Frank-Wolfe', type=int)

    args = parser.parse_args()
    
    N = args.N
    dpp_id = args.dpp_id
    k = args.k #cardinality constraint
    num_samples_mc = args.num_samples_mc #draw these many sets from x for multilinear relaxation
    num_fw_iter = args.num_fw_iter 
    if_herd = 0 
    torch_seed = args.torch_seed

    torch.manual_seed(torch_seed) 

    dpp_file = "/home/pankaj/Sampling/data/input/dpp/data/" + args.dpp_file

    (qualities, features) = read_dpp(dpp_file, 'dpp_' + str(dpp_id)) 

    dpp = DPP(qualities, features)

    call_FrankWolfe(dpp, args)

    print "Compeleted in " + str(time.clock() - tic) + 's'

if __name__ == '__main__':
    main()
