import sys
import numpy as np
import h5py

def clustered_DPP():
    #Quality same for all items
    #Feature vectors such that the items are grouped in k categories
    k = 20 
    n_dpp = 5
    N = 100 #number of items

    #Quality parameters
    mu_q = 2

    #Feature parameters
    D = 2*N #number of features
    mu_phi = 1 
    sigma_phi = 2
    
    params_string = '_'.join(str(x) for x in [N, mu_q, D, mu_phi, sigma_phi, n_dpp, k])
    hf = h5py.File('/home/pankaj/Sampling/data/input/dpp/data/clustered_dpp_' + params_string + '.h5', 'w')

    for p in range(5):
        q = np.array([mu_q]*N)
        
        phi = np.zeros((D, N))
        phi_k = sigma_phi * np.random.randn(D, k) + mu_phi #generate k very dissimilar feature vectors

        for i in range(N):
            category = np.random.randint(0, k) #pick a category at random
            phi_i = phi_k[:, category] + 0.01*np.random.randn(D) #add small noise to the category feature vector

            phi[:, i] = phi_i 

        #normalise feature vectors
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


def random_DPP():
    n_dpp = 5
    N = 100 #number of items
    #Quality parameters
    mu_q = 0.5 
    sigma_q = 0.5 

    #Feature parameters
    D = 2*N #number of features
    mu_phi = 0 
    sigma_phi = 0.1
    
    params_string = '_'.join(str(x) for x in [N, mu_q, sigma_q, D, mu_phi, sigma_phi, n_dpp])
    hf = h5py.File('/home/pankaj/Sampling/data/input/dpp/data/dpp_' + params_string + '.h5', 'w')

    for p in range(5):
        q = np.random.normal(mu_q, sigma_q, N)
        phi = np.random.randn(D, N)

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

if __name__ == '__main__':
    clustered_DPP()
