''' pca_nonlinear_mappings.py 
    All of the previous techniques worked best where the features were
    linearly separable, either totally separable for the Perceptron,
    or fairly separable for SVM, or at least the Principle Components were
    separable.
    
    When features are not linearly separable, non linear mappings may be
    used to separate the features.   
                
Created on Jul 2, 2016

from Python Machine Learning by Sebastian Raschka under the following license

The MIT License (MIT)

Copyright (c) 2015, 2016 SEBASTIAN RASCHKA (mail@sebastianraschka.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: richard lyman
'''
import numpy as np
import ocr_utils  
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh


def rbf_kernel_pca1(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]
        
    gamma: float
      Tuning parameter of the RBF kernel
        
    n_components: int
      Number of principal components to return

    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset   

    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack([eigvecs[:, -i] for i in range(1, n_components + 1)])

    return X_pc


from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y==0, 0], X[y==0, 1],
color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1],
color='blue', marker='o', alpha=0.5)
title='half_moon_1'
plt.title(title)
plt.tight_layout()
ocr_utils.show_figures(plt,title)


from sklearn.decomposition import PCA

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))

ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], 
            color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],
            color='blue', marker='o', alpha=0.5)
x1 = X_spca[y==0, 0]
x2 = X_spca[y==1, 0]
ax[1].scatter(x1, np.zeros((len(x1),1))+0.02, 
            color='red', marker='^', alpha=0.5)
ax[1].scatter(x2, np.zeros((len(x2),1))-0.02,
            color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
title='half_moon_2'

plt.tight_layout()
ocr_utils.show_figures(plt,title)


from matplotlib.ticker import FormatStrFormatter

X_kpca = rbf_kernel_pca1(X, gamma=15, n_components=2)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], 
            color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
            color='blue', marker='o', alpha=0.5)

x1 = X_spca[y==0, 0]
x2 = X_spca[y==1, 0]
ax[1].scatter(x1, np.zeros((len(x1),1))+0.02, 
            color='red', marker='^', alpha=0.5)
ax[1].scatter(x2, np.zeros((len(x2),1))-0.02,
            color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
title='half_moon_3'

plt.tight_layout()
ocr_utils.show_figures(plt,title)

from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
labels = np.unique(y)
plt.scatter(X[y==labels[0], 0], X[y==labels[0], 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==labels[1], 0], X[y==labels[1], 1], color='blue', marker='o', alpha=0.5)
title='circle 1'
plt.title(title)
plt.tight_layout()
ocr_utils.show_figures(plt,title)

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))

ax[0].scatter(X_spca[y==labels[0], 0], X_spca[y==labels[0], 1], 
            color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==labels[1], 0], X_spca[y==labels[1], 1],
            color='blue', marker='o', alpha=0.5)
x1 = X_spca[y==labels[0], 0]
x2 = X_spca[y==labels[1], 0]
ax[1].scatter(x1, np.zeros((len(x1),1))+0.02, 
            color='red', marker='^', alpha=0.5)
ax[1].scatter(x2, np.zeros((len(x2),1))-0.02,
            color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
title='circles_2'

plt.tight_layout()
ocr_utils.show_figures(plt, title)

X_kpca = rbf_kernel_pca1(X, gamma=15, n_components=2)
labels = np.unique(y)

x1 = X_kpca[y==labels[0], 0]
x2 = X_kpca[y==labels[1], 0]

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(x1, X_kpca[y==labels[0], 1], 
            color='red', marker='^', alpha=0.5)
ax[0].scatter(x2, X_kpca[y==labels[1], 1],
            color='blue', marker='o', alpha=0.5)

ax[1].scatter(x1, np.zeros((len(x1),1))+0.02, 
            color='red', marker='^', alpha=0.5)
ax[1].scatter(x2, np.zeros((len(x2),1))-0.02,
            color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
title= 'circles_3'

plt.tight_layout()
ocr_utils.show_figures(plt, title)



def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]
        
    gamma: float
      Tuning parameter of the RBF kernel
        
    n_components: int
      Number of principal components to return

    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset   
     
     lambdas: list
       Eigenvalues

    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack([eigvecs[:,-i] for i in range(1,n_components+1)])
    
    # Collect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1,n_components+1)]

    return alphas, lambdas


X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)

x_new = X[25]
x_new

x_proj = alphas[25] # original projection
x_proj

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

# projection of the "new" datapoint
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
x_reproj 

labels = np.unique(y)
x1 = alphas[y==labels[0], 0]
x2 = alphas[y==labels[1], 0]

plt.scatter(x1, np.zeros((len(x1))), 
            color='red', marker='^',alpha=0.5)
plt.scatter(x2, np.zeros((len(x2))), 
            color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black', label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green', label='remapped point X[25]', marker='x', s=500)
plt.legend(scatterpoints=1)
title= 'reproject'
plt.title(title)
plt.tight_layout()
ocr_utils.show_figures(plt, title)


from sklearn.decomposition import KernelPCA
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)
labels = np.unique(y)
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
title='KernelPCA using sklearn'
plt.title(title)
ocr_utils.show_figures(plt, title)


#     Using the apriori knowledge that the E13B characters were designed to
#     be read by scanning over the character and thus reading out the amount
#     of magnetized ink in a vertical column, we constructed computed features
#     of the column sums computed within the ocr_utils program.
#     
#     This worked fairly well providing a linearly separable data set, however,
#     some of the scanned E13B characters were scanned by a hand scanning,
#     during which the operator had titled the angle of the scanner so that
#     the character result was titled perhaps in the range +/- 10 degrees 
#     from the vertical.  The result is that a vertical column sum would
#     be incorrect for that character.
# import ocr_utils
# from sklearn.cross_validation import train_test_split
# 
# charsToTrain=(48,49)
# columnsXY = range(0,20)    
# charsToTrain=(48,51)
# columnsXY = (9,17)    
# y,X = ocr_utils.load_E13B(labels=charsToTrain , columns=columnsXY, nChars=300)  
# 
# feature_names = ['column {} sum'.format(columnsXY[i]) for i in range(len(columnsXY))]
#     
# X_train, X_test, y_train, y_test = \
#         train_test_split(X, y, test_size=0.3, random_state=0)
# 
# from sklearn.preprocessing import StandardScaler
# 
# sc = StandardScaler()
# X_train_std = sc.fit_transform(X_train)
# X_test_std = sc.fit_transform(X_test)
# 
# for gamma in range(1,20):
#     scikit_kpca = KernelPCA(n_components=2,  kernel="rbf", gamma=15)
#     X_skernpca = scikit_kpca.fit_transform(X_train_std)
#     labels = np.unique(y_train)
#     
#     plt.scatter(X_skernpca[y_train==labels[0], 0], X_skernpca[y_train==labels[0], 1], color='red', marker='^', alpha=0.5)
#     plt.scatter(X_skernpca[y_train==labels[1], 0], X_skernpca[y_train==labels[1], 1], color='blue', marker='o', alpha=0.5)
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     title='KernelPCA E13b gamma {}'.format(gamma)
#     plt.title(title)
#     ocr_utils.show_figures(plt, title)

print ('\n########################### No Errors ####################################')