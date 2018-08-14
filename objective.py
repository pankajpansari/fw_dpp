from subprocess import call
import os
import sys
import torch
from torch.autograd import Variable

eps = 1e-4

def submodObj(graph_file, sample):
    N = len(sample.data)
    this_set = [str(i) for i in range(N) if sample.data[i] == 1] 
    if this_set == []:
        return Variable(torch.Tensor([eps]))
    subset = ';'.join(this_set)
    subset = '"' + subset + '"'
    subset_len = len(this_set)
    influmax_command = './influmax/./influmax -n:' + graph_file +  ' -s:' + str(subset_len) + '-t:3 -b:4 -ls:' + str(subset)
    FNULL = open(os.devnull, 'w')
    call(influmax_command, stdout=FNULL, shell=True)
    f = open('./influence-info-network.txt', 'rU')
#    f = open(graph_file, 'rU')
    count = 0
    submodular_val = Variable(torch.Tensor([-1]))
    for line in f:
        count += 1;
        if count == 2:
            line.strip('\n')
            submodular_val = Variable(torch.Tensor([float(line)]))
            break
    
    return submodular_val 

def main():
    N = 8 
#    sample = Variable(torch.bernoulli(torch.FloatTensor([0.5]*N)))
    sample = Variable(torch.Tensor([0]*N))
#    graph_file = './influmax/example-network.txt'
    graph_file = '/home/pankaj/Sampling/data/working/06_05_2018/net1-network.txt'
    print sample
    print submodObj(graph_file, sample)

if __name__ == '__main__':
    main()
