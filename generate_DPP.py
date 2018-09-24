import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py

def orthogonal_DPPs():
    N = 20 
    n_clusters = 2 
    items_per_cluster = N/n_clusters
    D = 2*n_clusters
    epsilon = 1 
    epsilon_q = 0.1 

    hf = h5py.File('/home/pankaj/Sampling/data/input/dpp/data/orthogonal_dpp_N_' + str(N) + '_' + str(n_clusters) + '.h5', 'w')
    hf.create_group('quality')
    hf.create_group('feature')


    qualities = np.linspace(2, 2.9, 10)

#    qualities = [2]*5

    q = np.zeros(N)
    phi = np.zeros((D, N))
    phi_n = np.zeros((D, N))
    for k in range(n_clusters):
        phi_1 = np.zeros(D)
        phi_1[2*k:2*k + 2] = 1
        phi_1 = phi_1 + np.random.normal(0, 0.1, D)
#        phi_1[2*k] = 1 + epsilon
#        phi_1[2*k + 1] = 1 - epsilon

        phi_2 = np.zeros(D)
        phi_2[2*k:2*k + 2] = 1
        phi_2 = phi_2 + np.random.normal(0, 0.1, D)
#        phi_2[2*k] = 1 - epsilon
#        phi_2[2*k + 1] = 1 + epsilon
        
        q_1 = qualities[k]
        q_2 = qualities[k]
        q[2*k] = q_1 + epsilon_q
        q[2*k + 1] = q_2 - epsilon_q

        phi[:, 2*k] = phi_1
        phi[:, 2*k + 1] = phi_2

    for t in range(N):
        norm_constant = np.sqrt(phi[:, t].dot(phi[:, t]))
        if norm_constant != 0:
            phi_n[:, t] = phi[:, t]/norm_constant
        else:
            print "Cannot divide by 0!"


    hf['quality'].create_dataset('dpp_0', data = q)
    hf['feature'].create_dataset('dpp_0', data = phi_n)

    hf.close()

def gillenwater_DPPs():
   
    N = 20 
    S = np.random.randn(N, N)
    w_m_list = [0.5, 10, 1, 0.1, 0]
    u = np.random.randn(N)
    t = 0

    hf = h5py.File('/home/pankaj/Sampling/data/input/dpp/data/gillenwater_dpp_N_' + str(N) + '.h5', 'w')
    hf.create_group('quality')
    hf.create_group('feature')

    for w_m in w_m_list:
        quality = np.exp(w_m*u/2)
        hf['quality'].create_dataset('dpp_' + str(t), data = quality)
        hf['feature'].create_dataset('dpp_' + str(t), data = S)
    
        t = t + 1
    hf.close()

def gradually_hard_DPPs():

    n_dpp = 5
    N = 30 #number of items

    #Quality parameters
    mu_q = 3 
    sigma_q = 1

    #Feature parameters
    D = N #number of features
    mu_phi = 0
    sigma_phi = [0, 1, 2, 3] 
    
    params_string = '_'.join(str(x) for x in [N, mu_q, D, mu_phi, n_dpp])
    hf = h5py.File('/home/pankaj/Sampling/data/input/dpp/data/gradual_dpp_' + params_string + '.h5', 'w')
    hf.create_group('quality')
    hf.create_group('feature')

    for p in range(len(sigma_phi)):
        q = mu_q + sigma_q*np.random.randn(N)
        
        phi = sigma_phi[p] * np.random.randn(N, N)
        np.fill_diagonal(phi, 1)

        #normalise feature vectors
        phi_n = np.zeros((D, N))
        for t in range(N):
            norm_constant = np.sqrt(phi[:, t].dot(phi[:, t]))
            if norm_constant != 0:
                phi_n[:, t] = phi[:, t]/norm_constant
            else:
                print "Cannot divide by 0!"

        S = phi_n.dot(phi_n)
        plt.imshow(S, cmap = 'gray', vmin = -1, vmax = 1)
        plt.savefig('visualise_' + str(p) + '.png') 
        hf['quality'].create_dataset('dpp_' + str(p), data = q)
        hf['feature'].create_dataset('dpp_' + str(p), data = phi_n)
         
    hf.close()

def clustered_DPP():
    #Quality same for all items
    #Feature vectors such that the items are grouped in k categories
    k = 4 
    N = 20 #number of items
    D = N #number of features

    #Quality parameters
    mu_q = 3
    sigma_q = 1

    #Feature parameters
    mu_phi = 4 
    sigma_phi = 1
    
    params_string = '_'.join(str(x) for x in [N, D, k, mu_q, sigma_q, mu_phi, sigma_phi])
    hf = h5py.File('/home/pankaj/Sampling/data/input/dpp/data/clustered_dpp_' + params_string + '.h5', 'w')
    hf.create_group('quality')
    hf.create_group('feature')

    q_k = np.random.normal(mu_q, sigma_q, k)
    
    phi_k = np.random.normal(mu_q, sigma_q, (D, k))

    q = np.zeros(N)
    phi = np.zeros((D, N))

    for i in range(N):
        category = np.random.randint(0, k) #pick a category at random
        phi_i = phi_k[:, category] + 0.1*np.random.randn(D) #add small noise to the category feature vector

        q_i = q_k[category] + 0.1*np.random.randn()

        phi[:, i] = phi_i 
        q[i] = q_i 

    #normalise feature vectors
    normalise_factors = np.sum(phi, 0)
    phi_n = np.zeros((D, N))
    for t in range(N):
        norm_constant = np.sqrt(phi[:, t].dot(phi[:, t]))
        if norm_constant != 0:
            phi_n[:, t] = phi[:, t]/norm_constant
        else:
            print "Cannot divide by 0!"

    hf['quality'].create_dataset('dpp_0', data = q)
    hf['feature'].create_dataset('dpp_0', data = phi_n)

    hf.close()

def clustered_DPP_temp():
    #Quality same for all items
    #Feature vectors such that the items are grouped in k categories
    k = 5
    n_dpp = 5
    N = 5 #number of items

    #Quality parameters
    mu_q = 3 
    sigma_q = 1

    #Feature parameters
    mu_phi = 2 
    sigma_phi = 1 
    
    params_string = '_'.join(str(x) for x in [N, mu_q, mu_phi, sigma_phi, n_dpp, k])
    hf = h5py.File('/home/pankaj/Sampling/data/input/dpp/data/clustered_dpp_' + params_string + '.h5', 'w')
    hf.create_group('quality')
    hf.create_group('feature')

    for p in range(n_dpp):
        q = mu_q + sigma_q*np.random.randn(N)

        phi = np.zeros((N, N))
        phi_k = sigma_phi * np.random.randn(N, k) + mu_phi #generate k very dissimilar feature vectors

        for i in range(N):
            category = np.random.randint(0, k) #pick a category at random
            phi_i = phi_k[:, category] + (sigma_phi/10)*np.random.randn(N) #add small noise to the category feature vector

            phi[:, i] = phi_i 

        #normalise feature vectors
        phi_n = np.zeros((N, N))
        for t in range(N):
            norm_constant = np.sqrt(phi[:, t].dot(phi[:, t]))
            if norm_constant != 0:
                phi_n[:, t] = phi[:, t]/norm_constant
            else:
                print "Cannot divide by 0!"

        hf['quality'].create_dataset('dpp_' + str(p), data = q)
        hf['feature'].create_dataset('dpp_' + str(p), data = phi_n)
         
    hf.close()


def random_DPP():
    n_dpp = 5
    N = 10 #number of items
    D = N/2 #number of features

    #Quality parameters
    mu_q = 0.5 
    sigma_q = 0.5 

    #Feature parameters
    mu_phi = 0 
    sigma_phi = 0.1
    
    params_string = '_'.join(str(x) for x in [N, D, n_dpp, mu_q, sigma_q, mu_phi, sigma_phi])

    hf = h5py.File('/home/pankaj/Sampling/data/input/dpp/data/dpp_' + params_string + '.h5', 'w')

    for p in range(n_dpp):
        q = np.random.normal(mu_q, sigma_q, N)
        phi = np.random.normal(mu_phi, sigma_phi, (D, N))

        normalise_factors = np.sum(phi, 0)
        phi_n = np.zeros((D, N))
        for t in range(N):
            norm_constant = np.sqrt(phi[:, t].dot(phi[:, t]))
            if norm_constant != 0:
                phi_n[:, t] = phi[:, t]/norm_constant
            else:
                print "Cannot divide by 0!"

        B = np.zeros((D, N))
        for t in range(N):
            B[:, t] = q[t]*phi_n[:, t]
         
        L = (B.T).dot(B)
        hf.create_dataset('dpp_' + str(p), data = L)

    hf.close()

def param_search():
    mu_q = [0.1, 0.5, 1, 2, 5]
    sigma_q = [0.1, 0.5, 1, 2]

    mu_phi = [0.1, 0.5, 1, 2, 5]
    sigma_phi = [0.1, 0.5, 1, 2]

if __name__ == '__main__':
#    orthogonal_DPPs()
#    gillenwater_DPPs()
#    gradually_hard_DPPs()
    clustered_DPP()
#    random_DPP()
