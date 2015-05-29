'''
Created on May 26, 2015

@author: James
'''

#reimplementation + extension to stochastic variational inference of: https://github.com/wOOL/IBP

import numpy as np
import random
from inf.gen import genData
from inf.plot import plot_grid
from scipy.special import psi
import sys
import os
from inf.util import load_from_db
import time

       
np.random.seed(123)
random.seed(123)

#SYNTHETIC DATA
#generate data
# N = 500
# d1,d2 = 4,4
# D = d1*d2
# Kgrnd = 8
# grnd,X = genData(N,Kgrnd,d1,d2)
# plot_grid(grnd,d1,d2,title0='Ground Truth Means')

X = load_from_db(2000)
d1,d2 = 168,192
D = d1*d2
(N,D1) = np.shape(X)
assert D1==D
plot_grid(X[1:25,:],d1,d2,title0='Random Selection of Data',order='F',final=1)



#set hyperparams:
K = 15;
sigma_A = 3.0; #was 0.35
sigma_n = 1.0; #was 0.1
alpha = 1;

#init global r.v.'s
Phi = 0.1 * np.ones(K);
phi = np.random.uniform(0,1,(K,D));
tau = np.random.uniform(0,1,(K,2));

B = 100 #batch size
tau0 = 128 
kappa = 0.7 

for t in range(200):
    if t%100==0: print 'iteration',t
    #Sample a minibatch of data
    b = random.sample(range(N),B)

    #------- LOCAL UPDATES ---------
    print 'computing local updates...'
    #Update nu for minibatch (local updates)
    nu_local = np.random.uniform(0,1,(B,K));
    nu_local = nu_local/nu_local.sum(axis=1)[:,np.newaxis]
    #repeat updates to var params to z until convergence:
    v_old, diff, diff_thres = 0., 1., 1e-5
    t0 = 0
    for t0 in range(10):
#         t1=time.time()
        A = np.zeros((B,K))
        P = X[b,:]-np.dot(nu_local,phi)
        for k in range(K):
            A[:,k] = np.dot(phi[k,:], (P+np.outer(nu_local[:,k],phi[k,:])).T)
        v = -(psi(tau[:,0])-psi(tau[:,1])- 0.5/sigma_n**2*(D*Phi+(phi**2).sum(axis=1)) + 1/sigma_n**2*A)
        nu_local = 1./(1+np.exp(v))
#         t2 = time.time()
#         print 'fast',t2-t1
#         v1 = np.zeros((B,K))
#         for k in range(K):
#             v1[:,k] = -(psi(tau[k,0])-psi(tau[k,1])- 0.5/sigma_n**2*(D*Phi[k]+np.inner(phi[k,:],phi[k,:])) + \
#                             1/sigma_n**2*np.dot(phi[k,:], (X[b,:]-np.dot(nu_local,phi)+np.outer(nu_local[:,k],phi[k,:])).T))
#             #print 'v',v
# #         nu_local = 1./(1+np.exp(v1))
#         print 'slow',time.time() - t2
#         print 'v1',v1[:5,:]
#         print 'v',v[:5,:]
#         sys.exit(0)
        #check convergence:
        diff_v = (np.abs(v_old - v)).sum()
        v_old = v.copy()
        if diff_v<diff_thres: break
        t0+=1
    print 'converged after %i iters'%t0

    #------- GLOBAL UPDATES ---------
    
    # calculate intermediate Phi, phi (global updates given locals scaled up)
    Phi_im = 0.1 * np.ones(K)
    phi_im = np.random.uniform(0,1,(K,D))
    tau_im = np.random.uniform(0,1,(K,2))
    sf = N/B #scale factor to make gradients unbiased

    print 'computing global updates...'
    for k in range(K):
        Phi_im[k] = (1/(1/sigma_A**2 + (1/sigma_n**2) * sf * sum(nu_local[:,k]))) #scalar
        phi_im[k,:] = (1/sigma_n**2) * sf * np.dot(nu_local[:,k].T, X[b,:]-np.dot(nu_local,phi)+np.outer(nu_local[:,k],phi[k,:])) * Phi_im[k]

    # update tau
    nu_sum = sum(nu_local).T
    tau_im[:,0] = alpha/K + sf*nu_sum
    tau_im[:,1] = N + 1 - sf*nu_sum

    # perform gradient updates
    lr = (tau0 + t)**(-kappa)
    Phi = (1-lr)*Phi + lr*Phi_im
    phi = (1-lr)*phi + lr*phi_im
    tau = (1-lr)*tau + lr*tau_im


np.save('/Users/James/phi',phi)
plot_grid(phi,d1,d2,final=1,title0='Inferred Means',order='F')