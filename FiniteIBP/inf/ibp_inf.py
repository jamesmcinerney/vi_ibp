'''
Created on May 26, 2015

@author: James
'''

#reimplementation + extension to stochastic variational inference of: https://github.com/wOOL/IBP

import numpy as np
import random
from ibp.gen import genData
from ibp.plot import plot_grid
from scipy.special import psi
import sys
import os

       
np.random.seed(123)
random.seed(123)

#generate data
N = 10000
d1,d2 = 4,4
D = d1*d2
Kgrnd = 8
grnd,X = genData(N,Kgrnd,d1,d2)
plot_grid(grnd,d1,d2)
plot_grid(X[1:25,:],d1,d2)

#set hyperparams:
K = 15;
sigma_A = 0.35;
sigma_n = 0.1;
alpha = 1;

#init global r.v.'s
Phi = 0.1 * np.ones((D,D,K));
phi = np.random.uniform(0,1,(K,D));
tau = np.random.uniform(0,1,(K,2));

B = 3 #batch size
tau0 = 128 
kappa = 0.7 

for t in range(100):
    print 'iteration',t
    #Sample a minibatch of data
    b = random.sample(range(N),B)

    #------- LOCAL UPDATES ---------
    
    #Update nu for minibatch (local updates)
    nu_local = np.random.uniform(0,1,(B,K));
    nu_local = nu_local/nu_local.sum(axis=1)[:,np.newaxis]
    #repeat updates to var params to z until convergence:
    for t0 in range(10): #todo: include convergence check here
        for k in range(K):
            v = -(psi(tau[k,0])-psi(tau[k,1])- 0.5/sigma_n**2*(np.trace(Phi[:,:,k])+np.inner(phi[k,:],phi[k,:])) + \
                            1/sigma_n**2*np.dot(phi[k,:], (X[b,:]-np.dot(nu_local,phi)+np.outer(nu_local[:,k],phi[k,:])).T))
            #print 'v',v
            nu_local[:,k] = 1./(1+np.exp(v))

    #------- GLOBAL UPDATES ---------
    
    # calculate intermediate Phi, phi (global updates given locals scaled up)
    Phi_im = 0.1 * np.ones((D,D,K))
    phi_im = np.random.uniform(0,1,(K,D))
    tau_im = np.random.uniform(0,1,(K,2))
    sf = N/B #scale factor to make gradients unbiased

    for k in range(K):
        Phi_im[:,:,k] = (1/(1/sigma_A**2 + (1/sigma_n**2) * sf * sum(nu_local[:,k]))) * np.eye(D)
        phi_im[k,:] = (1/sigma_n**2) * sf * np.dot(nu_local[:,k].T, X[b,:]-np.dot(nu_local,phi)+np.outer(nu_local[:,k],phi[k,:])) * Phi_im[0,0,k]

    # update tau
    nu_sum = sum(nu_local).T
    tau_im[:,0] = alpha/K + sf*nu_sum
    tau_im[:,1] = N + 1 - sf*nu_sum

    # perform gradient updates
    lr = (tau0 + t)**(-kappa)
    Phi = (1-lr)*Phi + lr*Phi_im
    phi = (1-lr)*phi + lr*phi_im
    tau = (1-lr)*tau + lr*tau_im


plot_grid(phi,d1,d2,final=1)