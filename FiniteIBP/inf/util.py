'''
Created on May 28, 2015

@author: James
'''
import sqlite3
import numpy as np

def load_from_db(n_max=1e10):
    db_path = '/Users/James/Datasets/CroppedYale/yaleb.db'
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute('select text from corpus;')
    convert_image_str = lambda vstr: map(float, vstr[1:-1].split(','))
    xs = []
    for n,(v,) in enumerate(cur.fetchall()):
        #print 'v',v[:100]
        x = convert_image_str(v)
        #print 'x',x[:100]
        xs.append(x)
        if n%100==0: print 'loaded img',n
        if n>n_max: break
    X = np.array(xs)
    con.close()
    return X