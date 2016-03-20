# The current program computes three different errors (Frobenious norm,
# stress, variance difference) and the gain in storage for three
# dimensionality reduction methods (SVD, PCA, MDS). 
# We use as data the image "grumpy-cat.jpg"

from numpy import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from svd import svd
from pca import pca
from mds import mds

    
# load image
X = mpimg.imread('grumpy-cat.jpg', format='jpg')
X = X/255.0

# create separate matrices for each color R, G, B
Ro = X[:,:,0]
Go = X[:,:,1]
Bo = X[:,:,2]

# compute the storage space for X - image
storage_o = X.nbytes/1024.0/1024.0

index = array([[10,25,50,100,250]])

# initialize comparison metrics (Frobenious norm, stress, variance)
Fros = zeros((3,index.shape[1]))
var = zeros((3,index.shape[1]))
Stress = zeros((3,index.shape[1]))
storages = zeros((3,index.shape[1]))

# svd
tfros,tvars,tstr,mbs = svd(Ro,Go,Bo,index)
Fros[0,:] = transpose(tfros)
var[0,:] = transpose(tvars)
Stress[0,:] = transpose(tstr)
storages[0,:] = transpose(mbs)

# pca
tfros,tvars,tstr,mbs = pca(Ro,Go,Bo,index)
Fros[1,:] = transpose(tfros)
var[1,:] = transpose(tvars)
Stress[1,:] = transpose(tstr)
storages[1,:] = transpose(mbs)

# mds
tfros,tvars,tstr,mbs = mds(Ro,Go,Bo,index)
Fros[2,:] = transpose(tfros)
var[2,:] = transpose(tvars)
Stress[2,:] = transpose(tstr)
storages[2,:] = transpose(mbs)

storages = storages/storage_o

# plot images 
fig = plt.figure() 
# plot frobenius norm for the three methods
a = fig.add_subplot(221) 
plt.plot(log10(index[0]),Fros[0,:],color='r',linewidth=2.0,label='svd')
plt.plot(log10(index[0]),Fros[1,:],color='g',label='pca')
plt.plot(log10(index[0]),Fros[2,:],color='b',label='mds')
plt.xlabel('k log10 scale')
plt.ylabel('% Frobenius norm')
plt.legend(loc='upper right')
plt.draw()

# plot variance for the three methods
a = fig.add_subplot(222) 
plt.plot(log10(index[0]),var[0,:],color='r',linewidth=2.0,label='svd')
plt.plot(log10(index[0]),var[1,:],color='g',label='pca')
plt.plot(log10(index[0]),var[2,:],color='b',label='mds')
plt.xlabel('k log10 scale')
plt.ylabel('% cumulative variance')
plt.legend(loc='lower right')
plt.draw()

# plot stress for the three methods
a = fig.add_subplot(223) 
plt.plot(log10(index[0]),Stress[0,:],color='r',linewidth=2.0,label='svd')
plt.plot(log10(index[0]),Stress[1,:],color='g',label='pca')
plt.plot(log10(index[0]),Stress[2,:],color='b',label='mds')
plt.xlabel('k log10 scale')
plt.ylabel('% stress')
plt.legend(loc='upper right')
plt.draw()

# plot storage requerements for the three methods
a = fig.add_subplot(224) 
plt.plot(log10(index[0]),storages[0,:],color='r',linewidth=2.0,label='svd')
plt.plot(log10(index[0]),storages[1,:],color='g',label='pca')
plt.plot(log10(index[0]),storages[2,:],color='b',label='mds')
plt.xlabel('k log10 scale')
plt.ylabel('% Storage ratio')
plt.legend(loc='upper left')
plt.draw()

plt.show()
