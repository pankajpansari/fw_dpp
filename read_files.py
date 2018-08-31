import networkx as nx
import sys
import torch
import h5py
import numpy as np
from torch.autograd import Variable

def read_iterates(filename, N, num_iterates):

    f = open(filename, 'rU')
    x_list = []
    for t in range(num_iterates):
        x = []
        for i in range(N):
            val = float(next(f).strip('\n'))
            x.append(val)
        x_list.append(torch.Tensor(x))
        assert(next(f) == '\n')
    f.close()
    return x_list

def get_sfo_optimum(filename, N):

    x_good = Variable(torch.Tensor([0]*N))

    f = open(filename, 'rU')

    for _ in range(1):
        next(f)

    for line in f:
        num = int(line.strip('\n'))
        x_good[num] = 1

    return x_good

def get_fw_optimum(filename, N):

    x_good = Variable(torch.Tensor([0]*N))

    f = open(filename, 'rU')

    for _ in range(1):
        next(f)

    count = 0

    for line in f:
        x_good[count] = float(line.strip('\n'))
        count += 1

    return x_good

def read_dpp(dpp_file, N, dpp_id):
    with h5py.File(dpp_file, 'r') as hf:
        quality = torch.Tensor(hf['quality'].get(dpp_id))
        feature = torch.Tensor(hf['feature'].get(dpp_id))
        hf.close()
        return (quality, feature)


if __name__ == '__main__':
    N = 100 
    L = read_dpp("/home/pankaj/Sampling/data/input/dpp/data/dpp_100_0.5_0.5_200_0_0.1_5.h5", N, '/dpp_0')
    print L.shape 
