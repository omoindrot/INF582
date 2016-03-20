from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from numpy import genfromtxt
import csv

# Load the data set (wine). Variable data stores the final data (178 x 13)
my_data = genfromtxt('wine_data.csv', delimiter=',')
data = my_data[:, 1:]
target= my_data[:, 0]  # Class of each instance (1, 2 or 3)
print "Size of the data (rows, #attributes) ", data.shape


# Draw the data in 3/13 dimensions (Hint: experiment with different combinations of dimensions)
plt.figure(1)
plt.draw()
plt.scatter(data[:,1],data[:,2], c=target)
plt.xlabel('1st dimension')
plt.ylabel('2nd dimension')
plt.title("Vizualization of the dataset (2 out of 13 dimensions)")


#================= ADD YOUR CODE HERE ====================================
# Performs principal components analysis (PCA) on the n-by-p data matrix A (data)
# Rows of A correspond to observations (i.e., wines), columns to variables.
## TODO: Implement PCA
# Instructions: Perform PCA on the data matrix A to reduce its
#				dimensionality to 2 and 3. Save the projected
#				data in variables newData2 and newData3 respectively
#
# Note: To compute the eigenvalues and eigenvectors of a matrix
#		use the function eigval,eigvec = linalg.eig(M)
#

m, n = data.shape
print "data shape:", (m, n)
C = data - mean(data, axis=0)

W = C.T.dot(C)
w, U = linalg.eig(W)
U = U.argsort()

newData2 = C.dot(U[:, :2])
newData3 = C.dot(U[:, :3])


#=============================================================================


# Plot the first two principal components 
plt.figure(2)
plt.scatter(newData2[:,0],newData2[:,1], c=target)
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title("Projection to the top-2 Principal Components")
plt.draw()

# Plot the first three principal components 
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(newData3[:,0],newData3[:,1], newData3[:,2], c=target)
ax.set_xlabel('1st Principal Component')
ax.set_ylabel('2nd Principal Component')
ax.set_zlabel('3rd Principal Component')
ax.set_title("Projection to the top-3 Principal Components")
plt.show()  
