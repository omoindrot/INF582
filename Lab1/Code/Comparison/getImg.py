from numpy import *

def getImg(R, G, B):
    imgToSave = zeros((R.shape[0],R.shape[1],3))
	#the colors might have negative values  or values outside the valid range 
	#this is to be expected from numerical computations
	#so the absolute is necessary
	#the negative values are symmetrically  equivalent to their absolute
    R=absolute(R)
    G=absolute(G)
    B=absolute(B)
    R[R>1.0]=1.0
    G[G>1.0]=1.0
    B[B>1.0]=1.0
    imgToSave[:,:,0] = R
    imgToSave[:,:,1] = G
    imgToSave[:,:,2] = B
    return imgToSave

