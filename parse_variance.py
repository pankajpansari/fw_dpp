import os
import numpy as np
import sys
import matplotlib.pyplot as plt

coeff_list = [1e-05, 0.0001, 0.001, 0.01, 0.1]
nsamples_list = [1, 5, 10, 20, 50]

n_file = 5 

for coeff in coeff_list:
    for nsamples in nsamples_list:
        greedy = [0]*10
        fw = [0]*10
        for i in range(n_file):
            f = "/home/pankaj/Sampling/data/input/social_graphs/N_512/var_study/g_N_512_" + str(i) + "_20_" + str(nsamples) + "_0.4_100_0_" + str(coeff) + ".txt" 
            a = np.loadtxt(f)
            for t in range(10):
                b = a[3*t:3*(t+1), :]
                greedy[t] += (b[2,0] - b[0,0])
        greedy = [x/n_file for x in greedy]
        plt.plot(greedy, label = str(nsamples))
        plt.axhline(y = 0, linestyle = '--')
        plt.ylabel('Reduction in std deviation')
        plt.xlabel('Iterate')
        plt.xticks(range(1, 11))
    plt.legend()
    plt.savefig('var_greedy_plot_' + str(coeff) + '.jpg')
    plt.clf()

for coeff in coeff_list:
    for nsamples in nsamples_list:
        greedy = [0]*10
        fw = [0]*10
        for i in range(n_file):
            f = "/home/pankaj/Sampling/data/input/social_graphs/N_512/var_study/g_N_512_" + str(i) + "_20_" + str(nsamples) + "_0.4_100_0_" + str(coeff) + ".txt" 
            a = np.loadtxt(f)
            for t in range(10):
                b = a[3*t:3*(t+1), :]
                fw[t] += (b[2,0] - b[1,0])
        fw = [x/n_file for x in fw]
        plt.plot(range(1, 11), fw, label = str(nsamples))
        plt.axhline(y = 0, linestyle = '--')
        plt.ylabel('Reduction in std deviation')
        plt.xlabel('Iterate')
        plt.xticks(range(1, 11))
    plt.legend()
    plt.savefig('var_fw_plot_' + str(coeff) + '.jpg')
    plt.clf()

for coeff in coeff_list:
    for nsamples in nsamples_list:
        greedy = [0]*10
        fw = [0]*10
        for i in range(n_file):
            f = "/home/pankaj/Sampling/data/input/social_graphs/N_512/var_study/g_N_512_" + str(i) + "_20_" + str(nsamples) + "_0.4_100_0_" + str(coeff) + ".txt" 
            a = np.loadtxt(f)
            for t in range(10):
                b = a[3*t:3*(t+1), :]
                greedy[t] += (b[0,2] - b[0,1])
        greedy = [x/n_file for x in greedy]
        plt.plot(greedy, label = str(nsamples))
        plt.axhline(y = 0, linestyle = '--')
        plt.ylabel('Difference between true relaxation and estimated mean')
        plt.xlabel('Iterate')
        plt.xticks(range(1, 11))
    plt.legend()
    plt.savefig('mean_greedy_plot_' + str(coeff) + '.jpg')
    plt.clf()

for coeff in coeff_list:
    for nsamples in nsamples_list:
        greedy = [0]*10
        fw = [0]*10
        for i in range(n_file):
            f = "/home/pankaj/Sampling/data/input/social_graphs/N_512/var_study/g_N_512_" + str(i) + "_20_" + str(nsamples) + "_0.4_100_0_" + str(coeff) + ".txt" 
            a = np.loadtxt(f)
            for t in range(10):
                b = a[3*t:3*(t+1), :]
                fw[t] += (b[1,2] - b[1,1])
        fw = [x/n_file for x in fw]
        plt.plot(range(1, 11), fw, label = str(nsamples))
        plt.axhline(y = 0, linestyle = '--')
        plt.ylabel('Difference between true relaxation and estimated mean')
        plt.xlabel('Iterate')
        plt.xticks(range(1, 11))
    plt.legend()
    plt.savefig('mean_fw_plot_' + str(coeff) + '.jpg')
    plt.clf()


#for f in os.listdir("/home/pankaj/Sampling/data/input/social_graphs/N_512/var_study"):
#    if f != 'previous_results' and f != 'temp':
#        a = np.loadtxt("/home/pankaj/Sampling/data/input/social_graphs/N_512/var_study/" + f)
#        print f
#        print a

