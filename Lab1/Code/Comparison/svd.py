from numpy import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from frobenius import frobenius
from variance import variance
from stress import stress
from getImg import getImg

def svd(R,G,B,index):
    # svd the three colors
    Ur, Sr, Vr = linalg.svd(R)
    Ug, Sg, Vg = linalg.svd(G)
    Ub, Sb, Vb = linalg.svd(B)
    Sr = diag(Sr)
    Sg = diag(Sg)
    Sb = diag(Sb)
    Sr.resize((Ur.shape[0],Vr.shape[0]))
    Sg.resize((Ug.shape[0],Vg.shape[0]))
    Sb.resize((Ub.shape[0],Vb.shape[0]))
    
    frob = zeros((index.shape[1],1))
    var = zeros((index.shape[1],1))
    stre = zeros((index.shape[1],1))
    storg = zeros((index.shape[1],1))
    
    svdFig = plt.figure('SVD')
    nbImages = index.shape[1] + 1
    imcounter = 1
    a = svdFig.add_subplot(2,ceil(nbImages/2.0),imcounter) 
    imcounter += 1        
    X = getImg(R, G, B)
    plt.imshow(X)
    plt.title('true image')
    plt.axis('off')
    plt.show(block=False)

    for i in range(index.shape[1]):
        k = index[0,i]
        Urk = Ur[:,0:k]
        Srk = Sr[0:k,0:k]
        Vrk = Vr[:,0:k]
       
        # recontruction of the three color matrices based on the top -k elements
        Rk = dot(dot(Ur[:,0:k],Sr[0:k,0:k]),Vr[0:k,:])
        Gk = dot(dot(Ug[:,0:k],Sg[0:k,0:k]),Vg[0:k,:])
        Bk = dot(dot(Ub[:,0:k],Sb[0:k,0:k]),Vb[0:k,:])
        
        sizeR = Urk.nbytes/1024.0/1024.0
        sizeR = sizeR+Srk.nbytes/1024.0/1024.0
        sizeR = sizeR+Vrk.nbytes/1024.0/1024.0
        sizeG = Urk.nbytes/1024.0/1024.0
        sizeG = sizeG+Srk.nbytes/1024.0/1024.0
        sizeG = sizeG+Vrk.nbytes/1024.0/1024.0
        sizeB = Urk.nbytes/1024.0/1024.0
        sizeB = sizeB+Srk.nbytes/1024.0/1024.0
        sizeB = sizeB+Vrk.nbytes/1024.0/1024.0
        storg[i] = sizeR+sizeG+sizeB

        frob[i] = (frobenius(Rk,R)+frobenius(Gk,G)+frobenius(Bk,B))/3.0
        var[i] = real(variance(Rk,R)+variance(Gk,G)+variance(Bk,B))/3.0
        stre[i] = (stress(Rk,R)+stress(Gk,G)+stress(Bk,B))/3.0
        
        a = svdFig.add_subplot(2,ceil(nbImages/2.0),imcounter)
        imcounter += 1          
        X = getImg(Rk, Gk, Bk)
        plt.imshow(X)
        plt.title(str(index[0,i]) + ' dimensions')
        plt.axis('off')
        svdFig.canvas.draw()
         
    return frob, var, stre, storg
