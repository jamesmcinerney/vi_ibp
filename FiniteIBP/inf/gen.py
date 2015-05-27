'''
Created on May 26, 2015

@author: James
'''

import numpy as np

def genData(N,K,d1,d2):
    D = d1*d2
    
    #define some shapes
    s = np.zeros((K,D))
#     s[0,:] = np.array([[0,1,0],[0,1,0],[0,1,0]]).flatten()
#     s[1,:] = np.array([[1,1,1],[0,0,0],[0,0,0]]).flatten()
#     s[2,:] = np.array([[1,0,0],[0,1,0],[0,0,1]]).flatten()
#     s[3,:] = np.array([[0,0,0],[0,1,0],[0,0,0]]).flatten()
#     s[4,:] = np.array([[0,0,0],[0,1,1],[0,1,1]]).flatten()

    s[0,:] = np.array([[0,0,1,0],
                       [1,1,1,1],
                       [0,0,1,0],
                       [0,0,0,0]]).flatten()
    s[1,:] = np.array([[0,1,0,0],
                       [0,1,0,0],
                       [0,1,0,0],
                       [0,1,0,0]]).flatten()
    s[2,:] = np.array([[1,1,1,1],
                       [0,0,0,0],
                       [0,0,0,0],
                       [0,0,0,0]]).flatten()
    s[3,:] = np.array([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,1,0],
                       [0,0,0,1]]).flatten()
    s[4,:] = np.array([[0,0,0,0],
                       [0,0,0,0],
                       [1,1,0,0],
                       [1,1,0,0]]).flatten()
    s[5,:] = np.array([[1,1,1,1],
                       [1,0,0,1],
                       [1,0,0,1],
                       [1,1,1,1]]).flatten()
    s[6,:] = np.array([[0,0,0,0],
                       [0,1,1,0],
                       [0,1,1,0],
                       [0,0,0,0]]).flatten()
    s[7,:] = np.array([[0,0,0,1],
                       [0,0,0,1],
                       [0,0,0,1],
                       [0,0,0,1]]).flatten()

    pr_z = 0.3
    w = np.random.uniform(0.5,1.0,K) # weight of features
    mut= w[:,np.newaxis]*s
    z = np.random.binomial(1, pr_z, (N,K)) # each feature occurs with prob 0.3 independently 
    X = np.dot(z,mut)+np.random.randn(N,D)*0.1 

    return s,X