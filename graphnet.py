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

        num_node_feat = 3 + extra_feat_size
        self.num_edge_feat = 1
        self.t1 = nn.Parameter(torch.Tensor(self.p, num_node_feat))
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
        self.extra_feat_size = extra_feat_size

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.add_module(str(i), GraphConvLayer(p, w_std, extra_feat_size))

    def forward(self, x, adjacency, node_feat, edge_feat):

        batch_size = x.size(0)
        n_node = x.size(1)


        to_stack = []
        to_stack.append(Variable(torch.ones(batch_size, n_node)))
        to_stack.append(x)
        to_stack.append(1 - x)

#        for i in range(extra_feat.size(2)):
#            to_stack.append(extra_feat.select(2, i))

        node_feat = torch.stack(to_stack, 1).float()

        edge_feat = Variable(torch.ones(1, n_node, n_node))
        edge_feat = edge_feat.repeat(batch_size, 1, 1)

        mu = Variable(torch.zeros(batch_size, self.p, n_node))

        for layer in self.layers:
            mu = layer(node_feat, mu, adjacency, edge_feat)

        return mu

class GraphScorer(nn.Module):
    def __init__(self, p, w_std, k):
        super(GraphScorer, self).__init__()
        self.p = p
        self.w_std = w_std
        self.k = k

        self.t5_1 = nn.Parameter(torch.Tensor(1, self.p))
        self.t5_2 = nn.Parameter(torch.Tensor(1, self.p))
        self.t6 = nn.Parameter(torch.Tensor(self.p, self.p))
        self.t7 = nn.Parameter(torch.Tensor(self.p, self.p))

        self.reset()

    def reset(self):
        nn.init.normal_(self.t5_1, mean=0, std=self.w_std)
        nn.init.normal_(self.t5_2, mean=0, std=self.w_std)
        nn.init.normal_(self.t6, mean=0, std=self.w_std)
        nn.init.normal_(self.t7, mean=0, std=self.w_std)

    def forward(self, mu):
        accum = mu.sum(-1, keepdim=True)

        term1 = F.relu(self.t6.matmul(accum))
        term2 = F.relu(self.t7.matmul(mu))

        g_score = self.t5_1.matmul(term1).squeeze(1)
        per_node_score = self.t5_2.matmul(term2).squeeze(1)

        output = g_score.expand_as(per_node_score) + per_node_score

        output_distribution = torch.sigmoid(output)

#        factors = self.k/output_distribution.sum(dim = 1)
#        project_output = torch.mul(output_distribution, factors.expand_as(output_distribution.t()).t())
#        project_output[project_output > 1] = 1 - 1e-4
#        return project_output 
        return output_distribution

class MyNet(nn.Module):
    def __init__(self, k):
        super(MyNet, self).__init__()
        
        n_layer = 3
        p = 28
        w_scale = 1e-2
        extra_feat = 0
        self.conv = GraphConv(n_layer, p, w_scale, extra_feat)
        self.scorer = GraphScorer(p, w_scale, k)

    def forward(self, x, adjacency, node_feat, edge_feat):
        mu = self.conv(x, adjacency, node_feat, edge_feat)
        scores = self.scorer(mu)
        return scores

