import torch
import networkx as nx

from graphnet_temp import GraphConv, GraphScorer
from torch.autograd import Variable

N = 100
n_layer = 3
p = 32

# Create a sample
g = nx.erdos_renyi_graph(n = N, p = 0.15)
A_t = nx.adjacency_matrix(g).todense()
A_t = torch.from_numpy(A_t.astype("float32")).clone()

x_t = (torch.randn(N) >= 0).float()
w_t = A_t.clone()


net = GraphConv(n_layer, p, 0.1)
scorer = GraphScorer(p, 0.1)

# batch 2
x = Variable(x_t.unsqueeze(0).repeat(2, 1))
A = Variable(A_t.unsqueeze(0).repeat(2, 1, 1))
w = Variable(w_t.unsqueeze(0).repeat(2, 1, 1))

print x.size(), A.size(), w.size()
mu = net(x, A, w, None)
scores = scorer(mu)

