import numpy as np
import random
import sys
import os
from PIL import Image
import sqlite3

#np.seterr(all='raise') 
       
np.random.seed(123)
random.seed(123)

def normalize_img(db_path):
    #connect to db:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    #retrieve all data:
    cur.execute('select raw_text from corpus;')
    img_sum, img_sumsq = 0., 0.
    N = 0
    convert_image_str = lambda vstr: map(int, vstr[1:-1].split(','))
    for n,(vstr,) in enumerate(cur.fetchall()):
        xs = convert_image_str(vstr)
        img_sum += sum(xs)
        img_sumsq += sum(map(lambda x: int(x)**2, xs))
        if n%100==0: print 'n, img_sum, img_sumsq',n,img_sum,img_sumsq
        #if n>300: break
        N += 1
    #calculate mean and var
    D = len(xs)
    img_mean = img_sum / float(D*N)
    img_sd = np.sqrt( img_sumsq/float(D*N) - img_mean**2 )
    print 'img_mean, img_sd', img_mean, img_sd
    
    cur.execute('select doc_id,raw_text from corpus;')
    N = 0
    for (doc_id,vstr) in cur.fetchall():
        xs = np.array(convert_image_str(vstr))
        xs_n = (xs-img_mean)/img_sd
        cur.execute('update corpus set text = "%s" where doc_id = %i;' % (list(xs_n),doc_id))
        #if doc_id%100==0: print 'n, normalized xs',n,list(xs_n)
        N += 1
    con.commit()
    con.close()
    print 'done.'

if __name__=='__main__':
    normalize_img('/Users/James/Datasets/CroppedYale/yaleb.db')


    