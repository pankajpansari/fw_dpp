import torch
import networkx as nx
import sys
import numpy as np
import torch.optim as optim
import math
import os
import logger
import random
import fnmatch
import time

from graphnet import GraphConv, GraphScorer, MyNet
from torch.autograd import Variable
from influence import ic_model as submodObj
from variance import variance_estimate
from draw_graph import draw
import matplotlib.pyplot as plt
import visdom

random.seed(123)
np.random.seed(123)
torch.set_printoptions(precision = 2)
torch.manual_seed(123)

def reconstruction_loss(x, y):
    #Reconstruction loss - L2 difference between x and y 
    batch_size = x.size()[0]
    temp = x - y
    l2_norms = torch.norm(temp, 2, 1)

    return ((l2_norms**2).sum())/batch_size

def cross_entropy_loss(x, y):
    #Cross-entropy loss between x and y 
    batch_size = x.size()[0]
    eps = torch.zeros_like(y).fill_(1e-6)    #to avoid numerical issues in log
    term1 = (x * torch.log(y + eps)).sum()
    term2 = ((1 - x) * torch.log(1 - y + eps)).sum()

    return -(term1 + term2)/batch_size
    
def kl_divergence(x, y):
    #kl-divergence between x and y 
    batch_size = x.size()[0]
    eps = torch.zeros_like(y).fill_(1e-6)    #to avoid numerical issues in log
    term1 = (x * torch.log(y + eps)).sum()
    
    for t in range(y.shape[0]):
        assert(((1 - y[t] + eps[t]) <= 0).sum() == 0) 

    term2 = ((1 - x) * torch.log(1 - y + eps)).sum()
    term3 = (x * torch.log(x + eps)).sum()

    assert(torch.isnan(x).sum().item() == 0)
    assert(torch.isnan(y).sum().item() == 0)
    assert(torch.isnan(term1).sum().item() == 0)
    assert(torch.isnan(term2).sum().item() == 0)
    assert(torch.isnan(term3).sum().item() == 0)

    return -(term1 + term2 + term3)/batch_size
 
def get_data(graph_dir, nNodes, gt_param, train_size):

    file_list = os.listdir(graph_dir)
    data = []
    data_size = 1000
    for t in range(data_size):
        network_file = 'g_N_' + str(nNodes) + '_' + str(t) + '.txt'
        gt_file = 'g_N_' + str(nNodes) + '_' + str(t) + '_' + gt_param + '_gt.txt'
        if (network_file in file_list) and (gt_file in file_list):
            data.append((network_file, gt_file))

    ind_list1 = np.random.choice(range(data_size), size = train_size, replace = False)
    train_data = [data[ind] for ind in ind_list1]
    ind_list2 = [t for t in range(data_size) if t not in ind_list1]
    val_data = [data[ind] for ind in ind_list2]

    return [train_data, val_data]

def get_ground_truth(data_list, nNodes, graph_dir):

    ground_truth = Variable(torch.zeros(len(data_list), nNodes))
    for t in range(len(data_list)):
        f = open(graph_dir + data_list[t][1], 'rU')
        count = 0
        for line in f:
            ground_truth[t][count] = float(line)
            count += 1
        assert(torch.isnan(ground_truth[t]).sum().item() == 0)

    f.close()
    return ground_truth

def get_adjacency(data_list, nNodes, graph_dir):

    adjacency = Variable(torch.zeros(len(data_list), nNodes, nNodes))

    for t in range(len(data_list)):
        f = open(graph_dir + data_list[t][0], 'rU')
        adj = np.zeros((nNodes, nNodes))

        for line in f:
            if line.find('Nodes') != -1:
                N = int(line.split(' ')[2])
                assert(N == nNodes)
                break

        for _ in range(1):
            next(f)

        for line in f:
            from_id = int(line.split()[0])
            to_id = int(line.split()[1])
            adj[from_id, to_id] = 1

        f.close()

        adjacency[t] = Variable(torch.from_numpy(adj)).float()

    return adjacency

def get_node_feat(adjacency_list, nNodes):

    num_node_feat = 5
    node_feat = Variable(torch.zeros(len(adjacency_list), nNodes, num_node_feat))

    for t in range(len(adjacency_list)):

        G = nx.DiGraph(adjacency_list[t].numpy())
        
        in_degree = np.array((nx.in_degree_centrality(G)).values())
     
        out_degree = np.array((nx.out_degree_centrality(G)).values())
     
        closeness = np.array((nx.closeness_centrality(G)).values())
     
        between = np.array((nx.betweenness_centrality(G)).values())
     
        pagerank = np.array((nx.pagerank(G)).values())
     
        to_stack = [in_degree, out_degree, closeness, between, pagerank]
     
        assert(len(to_stack) == num_node_feat)
     
        node_feat[t] = Variable(torch.from_numpy(np.stack(to_stack, 1)).float())
 
    return node_feat
       
def round_x(x, N, k):

    assert(x.dim() == 2)
    assert(x.shape[1] == N)

    batch_size = x.shape[0]

    #Round the optimum solution and get function values
    rounded_x = torch.zeros_like(x)
    sorted_ind = torch.sort(x, dim = 1, descending = True)[1][:, 0:k]

    for i in range(batch_size):
        rounded_x[i, sorted_ind[i]] = 1

    return rounded_x

def get_submodular_ratio(net, adj, node_feat, gt_data, k, loss):

    data_size = adj.shape[0]
    nNodes = adj.shape[1]

    bs1 = torch.Tensor([float(k)/nNodes]*nNodes).repeat(data_size, 1)

    centrality = node_feat[:, :, 1]
    factors = 1.0/centrality.sum(dim = 1)
    bs2 = torch.mul(centrality, factors.expand_as(centrality.t()).t())
    
    bs1_loss = loss(gt_data, bs1)
    print "Baseline loss = ", bs1_loss.item()
    
    bs2_loss = loss(gt_data, bs2)
    print "Baseline 2 loss = ", bs2_loss.item()

    #Compare submodular values 
    net_output = net(adj, node_feat) 

    #Round and query submodular values for training set
    net_round = round_x(net_output, nNodes, k)
    gt_round = round_x(gt_data, nNodes, k)
    bs1_round = round_x(bs1, nNodes, k)
    bs2_round = round_x(bs2, nNodes, k)

    net_val = torch.zeros(data_size) 
    gt_val = torch.zeros(data_size) 
    bs1_val = torch.zeros(data_size) 
    bs2_val = torch.zeros(data_size) 

    for i in range(data_size):
        G = nx.DiGraph(adj[i].numpy())
        net_val[i] = submodObj(G, net_round[i], 0.5, 100)
        gt_val[i] = submodObj(G, gt_round[i], 0.5, 100)
        bs1_val[i] = submodObj(G, bs1_round[i], 0.5, 100)
        bs2_val[i] = submodObj(G, bs2_round[i], 0.5, 100)

    ratio1 = (net_val/gt_val).mean().item()
    ratio2 = (bs1_val/gt_val).mean().item()
    ratio3 = (bs2_val/gt_val).mean().item()

    print "Submodular function ratio (data): ", ratio1 
    print "Submodular function ratio (baseline 1): ", ratio2
    print "Submodular function ratio (baseline 2): ", ratio3

    return [ratio1, ratio2, ratio3]

def main():

    lr = float(sys.argv[1])
    mom = float(sys.argv[2])
    n_epochs = int(sys.argv[3])
    total_size = 1000
    train_size = int(sys.argv[4])
    val_size = total_size - train_size
    val_flag = int(sys.argv[5])
    k =  int(sys.argv[6])

    net = MyNet(k)

    for params in net.parameters():
        params.requires_grad = True

    loss = kl_divergence
    optimizer = optim.Adam(net.parameters(), lr=lr)
   
    save_dir = '/home/pankaj/Sampling/data/working/25_06_2018/'
    save_full_name = save_dir + 'training_lr_' + str(lr) + '_mom_' + str(mom) + '_tr_sz_' + str(train_size) + '_' + str(n_epochs)
    gt_param = '16_100_100_0.5_100'
    graph_dir = '/home/pankaj/Sampling/data/input/social_graphs/N_64/'

    [train_data, val_data] = get_data(graph_dir, 64, gt_param, train_size)
    
    train_adj = get_adjacency(train_data, 64, graph_dir)
    train_node_feat = get_node_feat(train_adj, 64)
    train_gt_data = get_ground_truth(train_data, 64, graph_dir)
    print "Got training data"

    if val_flag == 1:
        val_gt_data = get_ground_truth(val_data, 64, graph_dir)
        val_adj = get_adjacency(val_data, 64, graph_dir)
        val_node_feat = get_node_feat(val_adj, 64)
        centrality_val = val_node_feat[:, :, 1]
        print "Got validation data"

    tic = time.time()
    
    f = open(save_full_name + '_log.txt', 'w', 0)

    for epoch in range(0):
#    for epoch in range(n_epochs):
        #get minibatch
        optimizer.zero_grad()   # zero the gradient buffers
    
        train_output = net(train_adj, train_node_feat) 
    
        train_loss = loss(train_gt_data, train_output)
    
        train_loss.backward()
    
#        print net.scorer.t6.grad.data
#        sys.exit()

        optimizer.step()    # Does the update
    
        val_flag = 0
        if val_flag == 1:
            val_output = net(val_adj, val_node_feat) 
            val_loss = loss(val_gt_data, val_output)
    
            toc = time.time()
            print "Epoch: ", epoch, "       train loss = ", train_loss.item(), "      val loss = ", val_loss.item()
            f.write(str(epoch) + " " + str(train_loss.item()) + " " + str(val_loss.item()) + " " + str(toc - tic) + '\n')
        else:
            toc = time.time()
            print "Epoch: ", epoch, "       train loss = ", train_loss.item()
            f.write(str(epoch) + " " + str(train_loss.item()) +  " " + str(toc - tic) + '\n')
    
    
#    print "Training done.."
#    sys.exit()

#    torch.save(net, save_full_name + '.net')
    
    net = torch.load(save_full_name + '.net')

    train_output = net(train_adj, train_node_feat) 
    train_loss = loss(train_gt_data, train_output)
    print "Training loss = ", train_loss.item()


    if val_flag == 1:
        val_output = net(val_adj, val_node_feat) 
        val_loss = loss(val_gt_data, val_output)
        print "Validation loss = ", val_loss.item()


    #Test data
    graph_dir_test = '/home/pankaj/Sampling/data/input/social_graphs/N_32/'
    gt_param_test = '9_100_100_0.5_100'
    [test_data, temp] = get_data(graph_dir_test, 32, gt_param_test, 1000)
    test_adj = get_adjacency(test_data, 32, graph_dir_test)
    test_node_feat = get_node_feat(test_adj, 32)
    test_gt_data = get_ground_truth(test_data, 32, graph_dir_test)

    test_output = net(test_adj, test_node_feat) 
    test_loss = loss(test_gt_data, test_output)
    print "Test loss = ", test_loss.item()

    #########################
    print "Training data ratios"
    get_submodular_ratio(net, train_adj, train_node_feat, train_gt_data, k, loss)
    if val_flag == 1:
        print "Validation data ratios"
        get_submodular_ratio(net, val_adj, val_node_feat, val_gt_data, k, loss)
    print "Test data ratios"
    get_submodular_ratio(net, test_adj, test_node_feat, test_gt_data, 9, loss)

    f.close()

if __name__ == '__main__':
    main()
#    test_round()
