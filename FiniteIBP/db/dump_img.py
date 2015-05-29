


import numpy as np
import random
import sys
import os
from PIL import Image
import sqlite3

#np.seterr(all='raise') 
       
np.random.seed(123)
random.seed(123)

def dump_dat(db_path, image_root, max_N = 1000000):
    #connect to db:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    #create table:
    cur.execute('create table if not exists corpus (doc_id INTEGER PRIMARY KEY, raw_text TEXT, text TEXT);')
    
    #load image data:
    fpaths = [] #list of image paths for us to load
    for (dirpath,dirnames,filenames) in os.walk(image_root):
        for fn in filenames:
            if fn.endswith('.pgm') and not('Ambient' in fn): fpaths.append(dirpath+'/'+fn)
    #so now we have a list of image paths to process:
    print 'fpaths',fpaths[:10] #print selection of image paths
    
    #load files:
    N = 0
    for n,fn in enumerate(fpaths):
        if n%100==0: print 'image',n
        im = Image.open(fn) #.convert('LA') #load as grey image
        x = list(im.getdata())
        #x is an list of pixel intensities, dump it to db:
        cur.execute('insert into corpus (doc_id, raw_text, text) values (%i, "%s", null);' % (n, str(x)))
        N += 1
        if N>=max_N: break
    
    con.commit()
    con.close()

if __name__=='__main__':
    dump_dat('/Users/James/Datasets/CroppedYale/yaleb.db',
             '/Users/James/Datasets/CroppedYale/')