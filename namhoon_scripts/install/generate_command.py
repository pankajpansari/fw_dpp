from __future__ import print_function
import os
import time
import math
import sys
#Hyperparameters

def verify_variance_files():
    N = 512
    p = 0.4
    num_fw_iter = 20 
    nsamples_mlr_list = [1, 5, 10, 20, 50] 
    num_influ_iter = 100
    k = 20 #cardinality constraint
    a_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] 

    dirN = "/home/pankaj/Sampling/data/input/social_graphs/N_512/fw_opt/"
    opt_file_list = os.listdir(dirN)

    if_herd = 0
    for nsamples in [1, 5, 10, 20, 50]:
        for g_id in range(10):
            temp = 'g_N_' + str(N) + '_' + str(g_id) 
            seed = 123 + g_id
            opt_file = '_'.join(str(y) for y in [temp, k, nsamples, num_fw_iter, p, num_influ_iter, 0, 0, 0.0, seed]) + '.txt'
            if opt_file not in opt_file_list:
                print(opt_file)
#            if os.stat(dirN + opt_file).st_size == 0:
#                print(opt_file)

    sys.exit()
    for nsamples in nsamples_mlr_list:
        for i in range(10):
            for a in a_list:
                temp = 'g_N_' + str(N) + '_' + str(i) 

                var_file = '_'.join(str(y) for y in [temp, k, nsamples, p, num_influ_iter, if_herd, a]) + '.txt'
#                if var_file == '/home/pankaj/Sampling/data/input/social_graphs/N_512/var_study/g_N_512_6_20_1_0.4_100_0_0.01.txt':
#                    print(os.stat(var_file).st_size)
                if var_file not in var_file_list:
                    print(var_file)
                if os.stat(dirN + var_file).st_size == 0:
                    print(var_file)

def command_solution_variance():

    command_file = 'jobsQueued.txt'

    f = open(command_file, 'a')

    for nsamples in [1, 5, 10, 20]:
        for g_id in range(5):
            for t in range(5):
                seed = 123 + t 
                print('python ./cloned_repo/generate_ground_truth.py 512 ' + str(g_id) + ' 20 ' + str(nsamples) + ' 10 0.4 100 0 0 0.1 ' + str(seed), file = f)
                print('python ./cloned_repo/generate_ground_truth.py 512 ' + str(g_id) + ' 20 ' + str(nsamples) + ' 10 0.4 100 0 1 0.1 ' + str(seed), file = f)

def command_ground_truth():
    #N_list = [6, 7, 8, 9, 10] #in log terms (that is, graph has math.pow(2, N) nodes)
    N = 512
    p = 0.4
    num_fw_iter = 10 
    nsamples_mlr_list = [1, 5, 10, 20, 50] 
    num_influ_iter = 100
    k = 20 #cardinality constraint
    a_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] 

    command_file = 'jobsQueued.txt'

    f = open(command_file, 'w')

    for nsamples in nsamples_mlr_list:
        for i in range(5):
            for a in a_list:
                print('python ./cloned_repo/generate_ground_truth.py ' + ' '.join(str(x) for x in [N, i, k, nsamples, num_fw_iter, p, num_influ_iter, 0, 0, a]), file = f) 

#            for a in a_list:
#                #no herding
#                print('python ./cloned_repo/generate_ground_truth.py ' + ' '.join(str(x) for
#                    x in [N, i, k, nsamples, num_fw_iter, p, num_influ_iter, 0,
#                        1, 1, a]), file = f)
#
#                print('python ./cloned_repo/generate_ground_truth.py ' + ' '.join(str(x) for
#                    x in [N, i, k, nsamples, num_fw_iter, p, num_influ_iter, 0,
#                        1, 0, a]), file = f)

#                #herding
#                print 'python ./cloned_repo/generate_ground_truth.py ' + ' '.join(str(x) for x in [N, i, k, nsamples, num_fw_iter, p, num_influ_iter, 1, 1, 1, a])
#
#                print 'python ./cloned_repo/generate_ground_truth.py ' + ' '.join(str(x) for x in [N, i, k, nsamples, num_fw_iter, p, num_influ_iter, 1, 1, 0, a])
# 

    f.close()

#def command_variance():
#    id_list = range(5)
#    nsamples_list = [10, 100, 1000, 10000]
#    command_file = 'jobsQueued.txt'
#    f = open(command_file, 'w')
#    for nsamples in nsamples_list:
#        for i in id_list:
#           print 'python study_diffusion_param.py ' + str(i) + ' ' + str(nsamples)
#           f.write('python study_diffusion_param.py ' + str(i) + ' ' + str(nsamples) + '\n')
#
#def study_k_command():
#    id_list = range(5)
#    k_list = range(20, 300, 50)
#    nsamples = 10
#    command_file = 'jobsQueued.txt'
#    f = open(command_file, 'w')
#    for k in k_list:
#        for i in id_list:
#           print 'python study_diffusion_param.py ' + str(i) + ' ' + str(k)
#           f.write('python study_diffusion_param.py ' + str(i) + ' ' + str(k) + '\n')

if __name__ == '__main__':
#    study_k_command()
    command_solution_variance()
#    verify_variance_files()
