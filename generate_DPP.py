import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py

def random_DPP():

    n_dpp = 1
    N = 10 #number of items
    D = N #number of features

    #Quality parameters
    mu_q = 5 
    sigma_q = 3 

    #Feature parameters
    mu_phi = 0 
    sigma_phi = 0.1
    
    params_string = '_'.join(str(x) for x in [N, D, n_dpp, mu_q, sigma_q, mu_phi, sigma_phi])

    hf = h5py.File('/home/pankaj/Sampling/data/input/dpp/data/dpp_' + params_string + '.h5', 'w')
    hf.create_group('quality')
    hf.create_group('feature')

    for p in range(n_dpp):
        q = np.random.uniform(0, 1, N)
        phi = np.eye(N)

        phi_n = np.zeros((D, N))


        for t in range(N):
            norm_constant = np.sqrt(phi[:, t].dot(phi[:, t]))
            if norm_constant != 0:
                phi_n[:, t] = phi[:, t]/norm_constant
            else:
                print "Cannot divide by 0!"

        q[0] = 10
        q[1] = 15

        B = np.zeros((D, N))
        for t in range(N):
            B[:, t] = q[t]*phi_n[:, t]
         
        L = (B.T).dot(B)

        hf['quality'].create_dataset('dpp_0', data = q)
        hf['feature'].create_dataset('dpp_0', data = phi_n)

    hf.close()

if __name__ == '__main__':
    random_DPP()
