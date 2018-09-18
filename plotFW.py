#!/usr/bin/env python

from __future__ import division
# import modules used here -- sys is a very standard one
import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats 

def plotHistogram(a, start, end):

    x = a.flatten()
    print x.shape
    minVal = x.min()
    maxVal = x.max()
    print minVal
    print maxVal
    modeVal = stats.mode(x)[0]
    print modeVal 
    t = (end*minVal - start*maxVal)/(end - start)
    p = (maxVal - minVal)/(end - start)
    y = (x - t)/p
    transformy = np.tan((y + 0.5)*math.pi)
    transformy = np.log(y+start)
    plt.hist(transformy, 1000, log=True)

    plt.title("Negative Gradient Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
#    plt.savefig('gradient_histogram_transform.pdf')
    plt.show()
    
def plotHeatmap(a):
    print a.shape
    category = 1 
#    for category in range(21):
    b = a[category, :]
    b = b.reshape(320, 213) 
    print b.shape
    plt.imshow(b, cmap = 'hot', interpolation = 'nearest')
    plt.colorbar()
    plt.savefig('negGrad_heatmap_grass.pdf')
#    plt.show()

def linePlot(y, z, filename):
     plt.plot(y, linestyle='-', color = 'r', label = 'training') 
     plt.plot(z, linestyle='-', color = 'b', label = 'validation') 
     plt.xlabel('Iterations')
     plt.ylabel('KL-divergence')
     plt.legend()
     plt.show()

def linePlotShow(x):
     plt.plot(x, linestyle='-') 
     plt.xlabel('Iterations')
     plt.ylabel('Obj function val')
#     plt.ylim([0, 100])
#     plt.axhline(y[0], linewidth=2, color = 'b')
#     plt.xlim([-1, 1])
     plt.show()

def difference(a):
     b = a[a[:, 0] == 0].shape[0]
     diff = np.zeros(b)
     ngraphs = 2
     for i in range(ngraphs):
         ind = a[:, 0] == i 
         diff += a[ind, 2] - a[ind, 3]
     diff = diff/ngraphs
     for i in range(len(diff)):
         print i+1, diff[i]
     plt.plot(diff)
     plt.xlabel('Cardinality Constraint')
     plt.ylabel('Difference')
     plt.title('Plot of difference in submodular function values between optimum and best of 20 randomly drawn sets (averaged over 8 graphs of 32 nodes)')
     plt.savefig(sys.argv[2])

def plot_iterates_hist():
    filename = '/home/pankaj/Sampling/data/input/social_graphs/N_512/iterates/' + sys.argv[1]
    a = np.loadtxt(filename)
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        temp = a[512*i:512*(i + 1)]
        plt.hist(temp, bins = 100, range = (0, 1))
        plt.title("iter = " + str(i))
    plt.show()
#    plt.savefig(sys.argv[2])

def plot_solution_variance():

    solution_dir = '/home/pankaj/Sampling/data/input/social_graphs/N_512/fw_opt/'
    file_list = os.listdir(solution_dir)
    key_list = ['0_0_0.0', '0_0_0.1', '0_1_0.1']
    label_list = ['simple MC', 'convex with fw opt. 0.1', 'convex with greedy opt. 0.1']
    for p in range(3):
        d = []
        e = []
        nsamples_list = [1, 5, 10, 20]
        for nsamples in nsamples_list:
            a = []
            c = []
            for g_id in range(5):
                b = []
                for t in range(5):
                    seed = 123 + t 
                    opt_file = 'g_N_512_' + str(g_id) + '_20_' + str(nsamples) + '_20_0.4_100_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                    if opt_file not in file_list:
                        opt_file = 'g_N_512_' + str(g_id) + '_20_' + str(nsamples) + '_10_0.4_100_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                        if opt_file not in file_list:
                            print opt_file
                            sys.exit()

                    f = open(solution_dir + opt_file, 'r')
                    opt = float(next(f))
                    f.close()
                    b.append(opt)
                a.append(np.var(b))
                c.append(np.mean(b))
            d.append(np.mean(a))
            e.append(np.mean(c))
        print d, e
        plt.subplot(2, 1, 1)
        plt.plot(nsamples_list, d, label = label_list[p])
        plt.xlabel('# samples for relaxation estimation')
        plt.ylabel('Variance of 5 rounded solutions (averaged over 5 graphs)')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(nsamples_list, e, label = label_list[p])
        plt.xlabel('# samples for relaxation estimation')
        plt.ylabel('Mean of 5 rounded solutions (averaged over 5 graphs)')
        plt.legend()
#    plt.savefig('variance_solution.jpg')
    plt.show()

def plot_email_solution_variance():

    solution_dir = '/home/pankaj/Sampling/data/working/13-08-2018-email_sol_var_mc/workspace/'
    file_list = os.listdir(solution_dir)
    key_list = ['0_0_0.0']
    label_list = ['simple MC']
    for p in range(1):
        d = []
        e = []
        nsamples_list = [1, 5]
        for nsamples in nsamples_list:
            b = []
            for t in range(5):
                seed = 123 + t 
                opt_file = 'fw_opt_20_' + str(nsamples) + '_10_0.4_200_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                if opt_file not in file_list:
                    print opt_file
                    sys.exit()

                f = open(solution_dir + opt_file, 'r')
                opt = float(next(f))
                f.close()
                b.append(opt)
            d.append(np.var(b))
            e.append(np.mean(b))
        print d, e
        plt.subplot(2, 1, 1)
        plt.plot(nsamples_list, d, label = label_list[p], marker = '+', markersize = 10)
        plt.xlabel('# samples for relaxation estimation')
        plt.ylabel('Variance of 5 rounded solutions (1 graph, 10 FW iter)')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(nsamples_list, e, label = label_list[p], marker = '+', markersize = 10)
        plt.xlabel('# samples for relaxation estimation')
        plt.ylabel('Variance of 5 rounded solutions (1 graph, 10 FW iter)')
        plt.legend()
#    plt.savefig('variance_solution.jpg')
    plt.show()


def plot_DPP_solution_variance_ratio(k):
    greedy_sol = parse_DPP_greedy(k)

    solution_dir = '/home/pankaj/Sampling/data/working/20-08-2018-dpp_fw_variance/workspace/'
    file_list = os.listdir(solution_dir)
    key_list = '0_0_0.0'
    label_list = 'simple MC'
    fw_iter = 20
 
    nsamples_list = [1, 5, 10, 20, 50, 100]

    avg_mean = []
    avg_std = []
    for nsamples in nsamples_list:
        b = []
        c = []
        for dpp_id in range(5):
            greedy_val = greedy_sol[dpp_id]
            a = []
            for t in range(10):
                seed = 123 + t 
                opt_file = 'fw_opt_N_100_' + '_'.join(str(x) for x in [dpp_id, k, nsamples, fw_iter, key_list, seed]) + '.txt'

                if opt_file not in file_list:
                    print opt_file
                    sys.exit()

                f = open(solution_dir + opt_file, 'r')
                opt = float(next(f))
                f.close()
                a.append(opt*1.0/greedy_val) 
            b.append(np.mean(a))
            c.append(np.std(a))
        avg_mean.append(np.mean(b))
        avg_std.append(np.mean(c))

    print avg_mean
    print avg_std
    plt.errorbar(nsamples_list, avg_mean, yerr = avg_std, xerr = None, marker = '^')
    plt.axhline(1.0, c = 'r')
    plt.savefig('variance_ratio_solution_dpp_k_' + str(k) + '.jpg')
    plt.show()

def plot_DPP_solution_variance(k):

    solution_dir = '/home/pankaj/Sampling/data/working/20-08-2018-dpp_fw_variance/workspace/'
    file_list = os.listdir(solution_dir)
    key_list = '0_0_0.0'
    label_list = 'simple MC'
    fw_iter = 20
    a = []
    c = []
    nsamples_list = [1, 5, 10, 20, 50, 100]
    d = []
    e = []
    for nsamples in nsamples_list:
        a = []
        c = []
        for dpp_id in range(5):
            b = []
            for t in range(10):
                seed = 123 + t 
                opt_file = 'fw_opt_N_100_' + '_'.join(str(x) for x in [dpp_id, k, nsamples, fw_iter, key_list, seed]) + '.txt'

                if opt_file not in file_list:
                    print opt_file
                    sys.exit()

                f = open(solution_dir + opt_file, 'r')
                opt = float(next(f))
                f.close()
                b.append(opt)
            a.append(np.var(b))
            c.append(np.mean(b))
        d.append(np.mean(a))
        e.append(np.mean(c))
    print d, e

    #get average of greedy solutions
    greedy_val = parse_DPP_greedy()
    print greedy_val
    plt.subplot(2, 1, 1)
    plt.plot(nsamples_list, d, label = label_list, marker = '+', markersize = 10)
    plt.xlabel('# samples for relaxation estimation')
    plt.ylabel('Var of 10 rounded solutions (averaged over 5 DPPs)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(nsamples_list, e, label = label_list, marker = '+', markersize = 10)
    plt.axhline(greedy_val[2], c = 'r', ls = '--', label = 'greedy solution (avg)')
    plt.xlabel('# samples for relaxation estimation')
    plt.ylabel('Mean of 10 rounded solutions (averaged over 5 DPPs)')
    plt.legend()
    plt.savefig('variance_solution_dpp_k_' + str(k) + '.jpg')
    plt.show()

def parse_DPP_greedy(k):
    solution_dir = '/home/pankaj/Sampling/data/working/20-08-2018-dpp_fw_variance/greedy/'
    file_list = os.listdir(solution_dir)
    a = []
    for i in range(5):
        results_file = 'greedy_sfo_100_id_' + str(i) + '_k_'+ str(k) + '.txt'
        if results_file not in file_list:
            print results_file
            sys.exit()

        f = open(solution_dir + results_file, 'r')
        opt_val = float(next(f).split(' ')[0])
        f.close()
        a.append(opt_val)   
    return a

def parse_email_timing():

    solution_dir = '/home/pankaj/Sampling/data/working/13-08-2018-email_sol_var_mc/workspace/'
    file_list = os.listdir(solution_dir)
    key_list = ['0_0_0.0']
    label_list = ['simple MC']
    for p in range(1):
        d = []
        nsamples_list = [1, 5]
        for nsamples in nsamples_list:
            b = []
            for t in range(5):
                seed = 123 + t 
                opt_file = 'fw_log_20_' + str(nsamples) + '_10_0.4_200_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                if opt_file not in file_list:
                    print opt_file
                    sys.exit()

                f = open(solution_dir + opt_file, 'r')
                for i in range(9):
                    next(f)
                timing = float(next(f).split(' ')[0])
                f.close()
                print nsamples, timing
                b.append(timing)
            d.append(np.mean(b))
        print d

def plot_email_obj():

    solution_dir = '/home/pankaj/Sampling/data/working/13-08-2018-email_sol_var_mc/workspace/'
    file_list = os.listdir(solution_dir)
    key_list = ['0_0_0.0']
    label_list = ['simple MC']
    for p in range(1):
        nsamples_list = [5]
        for nsamples in nsamples_list:
            for t in range(5):
                b = []
                seed = 123 + t 
                opt_file = 'fw_log_20_' + str(nsamples) + '_10_0.4_200_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                if opt_file not in file_list:
                    print opt_file
                    sys.exit()

                f = open(solution_dir + opt_file, 'r')
                for i in range(10):
                    obj = float(next(f).split(' ')[1])
                    b.append(obj)
                f.close()
                plt.plot(range(1, 11), b, linestyle = '-')
    plt.xlabel('FW iterations')
    plt.ylabel('Relaxation objective')
    plt.show()

def plot_facebook_obj():

    solution_dir = '/home/pankaj/Sampling/data/working/13-08-2018-facebook_sol_var_mc/workspace/'
    file_list = os.listdir(solution_dir)
    key_list = ['0_0_0.0']
    label_list = ['simple MC']
    for p in range(1):
        nsamples_list = [1]
        for nsamples in nsamples_list:
            for t in range(5):
                b = []
                seed = 123 + t 
                opt_file = 'fw_log_20_' + str(nsamples) + '_10_0.4_200_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                if opt_file not in file_list:
                    print opt_file
                    sys.exit()

                f = open(solution_dir + opt_file, 'r')
                for i in range(4):
                    obj = float(next(f).split(' ')[1])
                    b.append(obj)
                f.close()
                plt.plot(range(1, 5), b, linestyle = '-')
    plt.xlabel('FW iterations')
    plt.ylabel('Relaxation objective')
    plt.show()


def parse_timing():

    solution_dir = '/home/pankaj/Sampling/data/input/social_graphs/N_512/fw_log/'
    file_list = os.listdir(solution_dir)
    key_list = ['0_0_0.0', '0_0_0.1', '0_1_0.1']
    label_list = ['simple MC', 'convex with fw opt. 0.1', 'convex with greedy opt. 0.1']
    for p in range(3):
        d = []
        e = []
        nsamples_list = [1, 5, 10, 20]
        for nsamples in nsamples_list:
            a = []
            c = []
            for g_id in range(5):
                b = []
                for t in range(5):
                    seed = 123 + t 
                    log_file = 'g_N_512_' + str(g_id) + '_20_' + str(nsamples) + '_20_0.4_100_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                    if log_file not in file_list:
                        log_file = 'g_N_512_' + str(g_id) + '_20_' + str(nsamples) + '_10_0.4_100_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                        if log_file not in file_list:
                            print log_file
                            sys.exit()

                    f = open(solution_dir + log_file, 'r')
                    for i in range(9):
                        next(f)
                    timing = float(next(f).split(' ')[0])
                    f.close()
                    b.append(timing)
                c.append(np.mean(b))
            e.append(np.mean(c))
        print e

def linePlotSave(x, filename):
     plt.plot(x, linestyle='-') 
     plt.xlabel('Iterations')
     plt.ylabel('Obj function val')
#     plt.ylim([0, 100])
#     plt.axhline(y[0], linewidth=2, color = 'b')
#     plt.xlim([-1, 1])
     plt.savefig(filename)
     plt.clf()

def plotRelaxationVariance(save_filename):

    file_list = ['dpp_0_5_0.001_0.001_0.9_0.9_0_800_1_1_100_train_variance.txt', 'dpp_1_5_0.001_0.001_0.9_0.9_0_1500_1_1_100_train_variance.txt', 'dpp_2_5_0.001_0.001_0.9_0.9_0_800_1_1_100_train_variance.txt', 'dpp_3_5_0.001_0.001_0.9_0.9_0_1500_1_1_100_train_variance.txt']

    for i in range(4): 
        filename = 'workspace/' + file_list[i] 
        a = np.loadtxt(filename)
        var_ratio = a[:, 2]/a[:, 1]
        plt.plot(a[:, 0], var_ratio, label = 'sigma = ' + str(i)) 
        plt.axes().set_xticks(a[:, 0])
    plt.xlabel('#samples for estimating relaxation')
    plt.ylabel('Learned proposal var/Nominal proposal var')
    plt.title('Variance of relaxation by importance sampling')
    plt.legend()
    save_filename = sys.argv[1]
    plt.savefig(save_filename)

def plot_x():
    x = [[ 0.4427,  0.8263,  0.0138,  0.0744,  0.2145,  0.2397,  0.1219,
          0.2239,  0.0032,  0.4859,  0.8425,  0.4935,  0.0657,  0.0629,
          0.4734,  0.2851,  0.4811,  0.0040,  0.3339,  0.7239,  0.5077,
          0.2950,  0.3651,  0.2326,  0.0964,  0.0688,  0.0593,  0.8338,
          0.5251,  0.3350]]
    y = [[ 0.4302,  0.7799,  0.0357,  0.0585,  0.1658,  0.2047,  0.1003,
          0.1756,  0.0218,  0.4741,  0.7489,  0.4862,  0.0505,  0.0677,
          0.4626,  0.2429,  0.4702,  0.0228,  0.3018,  0.7186,  0.5009,
          0.2566,  0.3396,  0.1917,  0.0674,  0.0574,  0.0446,  0.7941,
          0.5107,  0.3410]]
    plt.subplot(211)
    plt.imshow(x, cmap = 'gray', vmin = 0, vmax = 1)
    plt.subplot(212)
    plt.imshow(y, cmap = 'gray', vmin = 0, vmax = 1)
    plt.savefig('compare_x.png')

def plot_kl_multiple():

#     file_list = ['dpp_0_5_0.001_0.001_0.9_0.9_0_1500_1_1_100_exact_training_log.txt', 'dpp_0_5_0.001_0.001_0.9_0.9_0_1500_1_1_500_training_log.txt', 'dpp_0_5_0.001_0.001_0.9_0.9_0_1500_1_1_100_training_log.txt', 'dpp_0_5_0.001_0.001_0.9_0.9_0_1500_1_1_10_training_log.txt']
     file_list = ['dpp_0_5_0.001_0.001_0.9_0.9_0_1500_1_1_500_stochastic_training_log.txt', 'dpp_0_5_0.001_0.001_0.9_0.9_0_1500_1_1_100_stochastic_training_log.txt', 'dpp_0_5_0.001_0.001_0.9_0.9_0_1500_1_1_10_stochastic_training_log.txt']
     label_list = ['500 samples', '100 samples', '10 samples']
     for i in range(len(file_list)): 

         filename = 'workspace/' + file_list[i] 
         f = open(filename, 'r')

         val = []
         for line in f:
            a = line.split(' ')
            val.append(float(a[1]))

         plt.plot(val, linestyle='-', label = label_list[i]) 
     plt.xlabel('Iterations')
     plt.ylabel('Kl-based objective')
     plt.legend()
     save_filename = sys.argv[1]
     plt.savefig(save_filename)
 

# Gather our code in a main() function
def plot_kl_single():

     filename = 'workspace/' + sys.argv[1] 
     f = open(filename, 'r')

#         val = []
#         temp = -1
#         for line in f:
#           a = line.split(' ')
#            if int(a[0]) > temp:
#                temp = int(a[0])
#                val.append(float(a[1]))
#            else: 
#                break
#
#         plt.plot(val, linestyle='-') 
#         plt.xlabel('Iterations')
#         plt.ylabel('L2 reconstruction objective')
#         plt.savefig(save_filename)
#         plt.clf()

     val = []
     for line in f:
        a = line.split(' ')
        val.append(float(a[1]))

     plt.plot(val, linestyle='-') 
     plt.xlabel('Iterations')
     plt.ylabel('Kl-based objective')
#     plt.show()
     save_filename = sys.argv[2]
     plt.savefig(save_filename)
    

if __name__ == '__main__':
#    plot_x()
#    plotRelaxationVariance(sys.argv[1])
#   main()
#    plot_kl_multiple()
    plot_kl_single()

