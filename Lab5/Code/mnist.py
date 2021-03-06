# -*- coding: utf-8 -*-
"""

@author: jb
"""

import os, struct
from numpy import *
from array import array as pyarray

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Python function for importing the MNIST data set.
    """
    
    

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
        print "training"
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"


    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

   # ind = [ k for k in range(size) if lbl[k] in digits ]
   # N = len(ind)
    N=1
    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    #for i in range(len(ind)):
     #   images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
      #  labels[i] = lbl[ind[i]]



    return images, labels