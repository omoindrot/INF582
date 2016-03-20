from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

# Load the "gatlin" image data
X = loadtxt('gatlin.csv', delimiter=',')

#================= ADD YOUR CODE HERE ====================================
# Perform SVD decomposition
## TODO: Perform SVD on the X matrix
# Instructions: Perform SVD decomposition of matrix X. Save the 
#               three factors in variables U, S and V
#

m, n = X.shape
#returns (480, 640)

U, s, V = linalg.svd(X)
S = zeros((m,n))
for i in range(s.shape[0]):
    S[i, i] = s[i]

print S.shape
#=========================================================================

# Plot the original image
plt.figure(1)
plt.imshow(X, cmap = cm.Greys_r)
plt.title('Original image (rank 480)')
plt.axis('off')
plt.draw()

#================= ADD YOUR CODE HERE ====================================
# Matrix reconstruction using the top k = [10, 20, 50, 100, 200] singular values
## TODO: Create four matrices X10, X20, X50, X100, X200 for each low rank approximation
## using the top k = [10, 20, 50, 100, 200] singlular values 
#

def approx_with_k (U, S, V, k):
    U_k = U[:, :k]
    S_k = S[:k, :k]
    V_k = V[:k, :]
    return dot(U_k, dot(S_k,V_k))

X10 = approx_with_k(U, S, V, 10)
X20 = approx_with_k(U, S, V, 20)
X50 = approx_with_k(U, S, V, 50)
X100 = approx_with_k(U, S, V, 100)
X200 = approx_with_k(U, S, V, 200)

#=========================================================================



#================= ADD YOUR CODE HERE ====================================
# Error of approximation
## TODO: Compute and print the error of each low rank approximation of the matrix
# The Frobenius error can be computed as |X - X_k| / |X|
#

error10 = sum((X-X10)**2) / sum(X**2)
error20 = sum((X-X20)**2) / sum(X**2)
error50 = sum((X-X50)**2) / sum(X**2)
error100 = sum((X-X100)**2) / sum(X**2)
error200 = sum((X-X200)**2) / sum(X**2)

print error10
print error20
print error50
print error100
print error200


#=========================================================================



# Plot the optimal rank-k approximation for various values of k)
# Create a figure with 6 subfigures
plt.figure(2)

# Rank 10 approximation
plt.subplot(321)
plt.imshow(X10, cmap = cm.Greys_r)
plt.title('Best rank' + str(5) + ' approximation')
plt.axis('off')

# Rank 20 approximation
plt.subplot(322)
plt.imshow(X20,cmap = cm.Greys_r)
plt.title('Best rank' + str(20) + ' approximation')
plt.axis('off')

# Rank 50 approximation
plt.subplot(323)
plt.imshow(X50,cmap = cm.Greys_r)
plt.title('Best rank' + str(50) + ' approximation')
plt.axis('off')

# Rank 100 approximation
plt.subplot(324)
plt.imshow(X100,cmap = cm.Greys_r)
plt.title('Best rank' + str(100) + ' approximation')
plt.axis('off')

# Rank 200 approximation
plt.subplot(325)
plt.imshow(X200,cmap = cm.Greys_r)
plt.title('Best rank' + str(200) + ' approximation')
plt.axis('off')

# Original
plt.subplot(326)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (Rank 480)')
plt.axis('off')

plt.draw()


#================= ADD YOUR CODE HERE ====================================
# Plot the singular values of the original matrix
## TODO: Plot the singular values of X versus their rank k

plt.figure(3)
plt.plot(s)
plt.draw()


#=========================================================================

plt.show() 

