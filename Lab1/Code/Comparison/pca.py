from numpy import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from frobenius import frobenius
from variance import variance
from stress import stress
from getImg import getImg

def pca(R,G,B,index):
    frob = zeros((index.shape[1],1))
    var = zeros((index.shape[1],1))
    stre = zeros((index.shape[1],1))
    storg = zeros((index.shape[1],1))
    
    pcaFig = plt.figure('PCA') 
    nbImages = index.shape[1] + 1
    imcounter = 1
    a = pcaFig.add_subplot(2,ceil(nbImages/2.0),imcounter) 
    imcounter += 1        
    X = getImg(R, G, B)
    plt.imshow(X)
    plt.title('true image')
    plt.axis('off')
    plt.show(block=False)
    
    rows = R.shape[0]
    columns = R.shape[1]
    
    mnr = mean(R,0)   
    mnr = mnr.reshape(1,len(mnr))
    newR = R - dot(ones((rows,1)),mnr) # centralize
    Rc = dot(transpose(newR),newR)/(rows-1) # covariance
    Ur, Sr, Vr = linalg.svd(Rc) # principacl components
    
    mng = mean(G,0)
    mng = mng.reshape(1,len(mng))
    newG = G - dot(ones((rows,1)),mng) # centralize
    Gc = dot(transpose(newG),newG)/(rows-1) # covariance
    Ug, Sg, Vg = linalg.svd(Gc) # principacl components
    
    mnb = mean(B,0)
    mnb = mnb.reshape(1,len(mnb))
    newB = B - dot(ones((rows,1)),mnb) # centralize
    Bc = dot(transpose(newB),newB)/(rows-1) # covariance
    Ub, Sb, Vb = linalg.svd(Bc) # principacl components
    
    for i in range(index.shape[1]):
        k = index[0,i]
        
        Urk = Ur[:,0:k]
        # project R to k principal componets
        Rk = dot(newR,Ur[:,0:k]) # we need to store k colums (projection)
        sizeR = Rk.nbytes/1024.0/1024.0
        # recontruct imgae in the original dimensions 
        Rk = dot(Rk,transpose(Ur[:,0:k])) # we need to store k rows (components)
        Rk = Rk + tile(mnr,(rows,1)) # and we need to store the mean
        sizeR = sizeR+Urk.nbytes/1024.0/1024.0
        sizeR = sizeR+mnr.nbytes/1024.0/1024.0
        
        Ugk = Ug[:,0:k]
        # project R to k principal componets
        Gk = dot(newG,Ug[:,0:k]) # we need to store k colums (projection)
        sizeG = Gk.nbytes/1024.0/1024.0
        # recontruct imgae in the original dimensions 
        Gk = dot(Gk,transpose(Ug[:,0:k])) # we need to store k rows (components)
        Gk = Gk + tile(mng,(rows,1)) # and we need to store the mean
        sizeG = sizeG+Ugk.nbytes/1024.0/1024.0
        sizeG = sizeG+mng.nbytes/1024.0/1024.0
        
        Ubk = Ub[:,0:k]
        # project R to k principal componets
        Bk = dot(newB,Ub[:,0:k]) # we need to store k colums (projection)
        sizeB = Bk.nbytes/1024.0/1024.0
        # recontruct imgae in the original dimensions 
        Bk = dot(Bk,transpose(Ub[:,0:k])) # we need to store k rows (components)
        Bk = Bk + tile(mnb,(rows,1)) # and we need to store the mean
        sizeB = sizeB+Ubk.nbytes/1024.0/1024.0
        sizeB = sizeB+mnb.nbytes/1024.0/1024.0
        
        storg[i] = sizeR+sizeG+sizeB
        frob[i] = (frobenius(Rk,R)+frobenius(Gk,G)+frobenius(Bk,B))/3.0
        var[i] = real(variance(Rk,R)+variance(Gk,G)+variance(Bk,B))/3.0
        stre[i] = (stress(Rk,R)+stress(Gk,G)+stress(Bk,B))/3.0
        
        a = pcaFig.add_subplot(2,ceil(nbImages/2.0),imcounter)
        imcounter += 1          
        X = getImg(Rk, Gk, Bk)
        plt.imshow(X)
        plt.title(str(index[0,i]) + ' dimensions')
        plt.axis('off')
        pcaFig.canvas.draw()
             
    return frob, var, stre, storg
