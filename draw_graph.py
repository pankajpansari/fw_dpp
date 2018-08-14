import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
import visdom

wdir = "/home/pankaj/Sampling/data/input/social_graphs/"

def draw(G, gt_values, network_values, save_prefix):

    pos = nx.random_layout(G)

    nx.draw(G, pos = pos, with_labels = True, cmap=plt.cm.Blues, node_color=
            gt_values, alpha = 0.8)
    plt.draw()
    plt.savefig(save_prefix.replace('.txt', '_gt.png'))
    plt.clf()

    nx.draw(G, pos = pos, with_labels = True, cmap=plt.cm.Blues,
            node_color=network_values, alpha = 0.8)
    plt.draw()
    plt.savefig(save_prefix.replace('.txt', '_net.png'))
    plt.clf()

    baseline_values = ['lightblue' for node in G.nodes()]
    nx.draw(G, pos = pos, with_labels = True, cmap=plt.cm.Blues,
            node_color=baseline_values, alpha = 0.8)
    plt.draw()
    plt.savefig(save_prefix.replace('.txt', '_bl0.png'))
    plt.clf()    

def main():
    draw()

if __name__ == '__main__':
    main()
