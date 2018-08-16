import sys
import numpy as np
import h5py

def main():
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
    main()
