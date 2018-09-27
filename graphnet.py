import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

class GraphConvLayer(nn.Module):
    def __init__(self, p, w_std, extra_feat_size=0):
        super(GraphConvLayer, self).__init__()
        self.p = p
        self.w_std = w_std

        self.num_node_feat = 4 
        self.num_edge_feat = 2 
        self.t1 = nn.Parameter(torch.Tensor(self.p, self.num_node_feat))
        self.t2 = nn.Parameter(torch.Tensor(self.p, self.p))
        self.t3 = nn.Parameter(torch.Tensor(self.p, self.p))
        self.t4 = nn.Parameter(torch.Tensor(self.p, self.num_edge_feat))

        self.reset()

    def reset(self):
        nn.init.normal_(self.t1, mean=0, std=self.w_std)
        nn.init.normal_(self.t2, mean=0, std=self.w_std)
        nn.init.normal_(self.t3, mean=0, std=self.w_std)
        nn.init.normal_(self.t4, mean=0, std=self.w_std)

    def forward(self, node_feat, mu, adjacency, edge_feat):
        batch_size = node_feat.size(0)
        n_node = adjacency.size(1)
        term1 = self.t1.matmul(node_feat)
        term2 = self.t2.matmul(mu).matmul(adjacency)
        term3_1 = F.relu(self.t4.matmul(edge_feat.view(batch_size, self.num_edge_feat, n_node * n_node)))
        term3_1 = term3_1.view(batch_size, self.p, n_node, n_node).sum(-1)
        term3 = self.t3.matmul(term3_1)
        
        new_mu = F.relu(term1 + term2 + term3)

        return new_mu

class GraphConv(nn.Module):
    def __init__(self, n_layer, p, w_std, extra_feat_size=0):
        super(GraphConv, self).__init__()
        self.p = p

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.add_module(str(i), GraphConvLayer(p, w_std))

    def forward(self, x, adjacency, node_feat, edge_feat):

        batch_size = x.size(0)
        n_node = x.size(1)


        #Add bias and x to node features
        to_stack = []
        bias = torch.ones(batch_size, 1, n_node)
        neg_bias = -1*torch.ones(batch_size, 1, n_node)
        temp = node_feat.repeat(batch_size, 1, 1) 
        augmented_node_feat = torch.cat((bias, neg_bias, torch.unsqueeze(x, 1), temp), 1) 

        #Add bias to the edge features
        bias = torch.ones(batch_size, 1, n_node, n_node)
        temp = edge_feat.repeat(batch_size, 1, 1, 1) 
        augmented_edge_feat = torch.cat((bias, temp), 1) 

        adjacency = adjacency.repeat(batch_size, 1, 1)
        mu = Variable(torch.zeros(batch_size, self.p, n_node))

        for layer in self.layers:
            mu = layer(augmented_node_feat, mu, adjacency, augmented_edge_feat)

        return mu

class GraphScorer(nn.Module):
    def __init__(self, p, w_std, rank):
        super(GraphScorer, self).__init__()
        self.p = p
        self.w_std = w_std
        self.rank = rank

        self.t5_1 = nn.Parameter(torch.Tensor(1, self.p))
        self.t5_2 = nn.Parameter(torch.Tensor(1, self.p))
        self.t6 = nn.Parameter(torch.Tensor(self.p, self.p))
        self.t7 = nn.Parameter(torch.Tensor(self.p, self.p))
        self.t8_1 = nn.Parameter(torch.Tensor(self.rank, self.p))
        self.t8_2 = nn.Parameter(torch.Tensor(self.rank, self.p))

        self.reset()

    def reset(self):
        nn.init.normal_(self.t5_1, mean=0, std=self.w_std)
        nn.init.normal_(self.t5_2, mean=0, std=self.w_std)
        nn.init.normal_(self.t6, mean=0, std=self.w_std)
        nn.init.normal_(self.t7, mean=0, std=self.w_std)
        nn.init.normal_(self.t8_1, mean=0, std=self.w_std)
        nn.init.normal_(self.t8_2, mean=0, std=self.w_std)

    def forward(self, mu):
        accum = mu.sum(-1, keepdim=True)

        term1 = F.relu(self.t6.matmul(accum))
        term2 = F.relu(self.t7.matmul(mu))
        print mu.size()
        g_score = self.t5_1.matmul(term1).squeeze(1)
        per_node_score = self.t5_2.matmul(term2).squeeze(1)

        output = g_score.expand_as(per_node_score) + per_node_score

        output_marg = torch.sigmoid(output)

        g_score_cov = self.t8_1.matmul(term1).squeeze(1)
        per_node_score_cov = self.t8_2.matmul(term2).squeeze(1)

#        output_cov = g_score_cov.expand_as(per_node_score_cov) + per_node_score_cov
        output_cov = mu[:, 0:5, :]
#        print output_cov.size()
#        sys.exit()
        return [output_marg, output_cov]

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        
        n_layer = 3
        p = 28 
        w_scale = 1e-2
        rank = 5 
        self.conv = GraphConv(n_layer, p, w_scale)
        self.scorer = GraphScorer(p, w_scale, rank)

    def forward(self, x, adjacency, node_feat, edge_feat):
        mu = self.conv(x, adjacency, node_feat, edge_feat)
        scores = self.scorer(mu)
        return scores

