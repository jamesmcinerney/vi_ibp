'''
Created on May 26, 2015

@author: James
'''

from matplotlib.pyplot import *
from matplotlib import cm
import numpy as np

def plot_grid(X,d1,d2,final=0,title0=''):
    #takes a NxD matrix and plots N d1xd2 images (where d1xd2==D)
    (N,D) = np.shape(X)
    assert d1*d2==D,'%i %i %i'%(d1,d2,D)
    n1 = int(np.sqrt(N))+1
    for n in range(N):
        pl = subplot(n1,n1,n)
        x = np.reshape(X[n,:],(d1,d2))
        imgplot = pl.imshow(x, cmap=cm.Greys_r)
        imgplot.set_interpolation('nearest')
    title(title)
    if final: show()
    else: draw(); figure()