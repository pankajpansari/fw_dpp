import networkx as nx
import sys
import torch
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

if __name__ == '__main__':
    N = 4039 
    G = read_facebook_graph('/home/pankaj/Sampling/data/input/social_graphs/facebook/facebook_combined.txt', N)
    print G.number_of_edges()
