from numpy import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from frobenius import frobenius
from variance import variance
from stress import stress
from getImg import getImg

def mds(R,G,B,index):
    frob = zeros((index.shape[1],1))
    var = zeros((index.shape[1],1))
    stre = zeros((index.shape[1],1))
    storg = zeros((index.shape[1],1))
    
    mdsFig = plt.figure('MDS') 
    nbImages = index.shape[1] + 1
    imcounter = 1
    a = mdsFig.add_subplot(2,ceil(nbImages/2.0),imcounter) 
    imcounter += 1        
    X = getImg(R, G, B)
    plt.imshow(X)
    plt.title('true image')
    plt.axis('off')
    plt.show(block=False)

    Rsim = dot(R,transpose(R))
    Ur, Sr, Vr = linalg.svd(Rsim)

    Gsim = dot(G,transpose(G))
    Ug, Sg, Vg = linalg.svd(Gsim)

    Bsim = dot(B,transpose(B))
    Ub, Sb, Vb = linalg.svd(Bsim)

    # in MDS we keed the distance space so reconstructing the original data is
    # not possible....but we can project the data to the new distance space
    # this is the same as pca
    # in fact the ain difference is the lack of centering

    for i in range(index.shape[1]):
        k = index[0,i]
        
        Urk = Ur[:,0:k]
        Rk = dot(transpose(Ur[:,0:k]),R)
        sizeR = Rk.nbytes/1024.0/1024.0
        Rk = dot(Ur[:,0:k],Rk)
        sizeR = sizeR+Urk.nbytes/1024.0/1024.0
        
        Ugk = Ug[:,0:k]
        Gk = dot(transpose(Ug[:,0:k]),G)
        sizeG = Gk.nbytes/1024.0/1024.0
        Gk = dot(Ug[:,0:k],Gk)
        sizeG = sizeG+Ugk.nbytes/1024.0/1024.0
        
        Ubk = Ub[:,0:k]
        Bk = dot(transpose(Ub[:,0:k]),B)
        sizeB = Bk.nbytes/1024.0/1024.0
        Bk = dot(Ub[:,0:k],Bk)
        sizeB = sizeB+Ubk.nbytes/1024.0/1024.0
 
        storg[i] = sizeR+sizeG+sizeB
        frob[i] = (frobenius(Rk,R)+frobenius(Gk,G)+frobenius(Bk,B))/3.0
        var[i] = real(variance(Rk,R)+variance(Gk,G)+variance(Bk,B))/3.0
        stre[i] = (stress(Rk,R)+stress(Gk,G)+stress(Bk,B))/3.0
        
        a = mdsFig.add_subplot(2,ceil(nbImages/2.0),imcounter)
        imcounter += 1          
        X = getImg(Rk, Gk, Bk)
        plt.imshow(X)
        plt.title(str(index[0,i]) + ' dimensions')
        plt.axis('off')
        mdsFig.canvas.draw()
    
    return frob, var, stre, storg  
