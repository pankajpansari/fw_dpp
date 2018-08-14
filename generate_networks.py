from subprocess import call
import os
import math
import sys
import numpy as np

seed_m = np.array([[0.8, 0.7], [0.5, 0.3]])
seed_text = '"' + str(seed_m[0, 0]) + ' ' + str(seed_m[0, 1]) + '; ' + str(seed_m[1, 0]) + ' ' + str(seed_m[1, 1]) + '"'
seed_value = 123
num_graphs = 1000 
logN_list = [9]

FNULL = open(os.devnull, 'w')
for logN in logN_list: 
#    destination_dir = '/home/pankaj/Sampling/data/input/social_graphs/k_' + str(k) + '/'
    n_nodes = int(math.pow(2, logN))
    n_edges = int(math.pow(np.sum(seed_m), logN))
    destination_dir = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(n_nodes) + '/'
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    print "Generating " + str(num_graphs) + " kronecker graphs with #nodes: " + str(n_nodes) + " #edges: " + str(n_edges)
    for t in range(num_graphs):
        if t % 100 == 0:
            print ".", 
        filename = destination_dir + "g_N_" + str(n_nodes) + '_' + str(t) + '.txt'
        seed_value = seed_value + 1
#        command = './influmax/./generate_nets -t:0 -g:' + seed_text + ' -n:' + str(n_nodes) + ' -e:' + str(n_edges) + ' -f:' + filename
        command = '/home/pankaj/Sampling/code/social_networks/snap/examples/krongen/krongen -m:' + seed_text + ' -i:' + str(logN) + ' -o:' + filename + ' -s:' + str(seed_value)
#        print command
        call(command, stdout=FNULL, shell=True)
    print "Done"

