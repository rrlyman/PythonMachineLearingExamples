''' principal_component_analysis.py 
Principal Component Analysis reduces the dimensionality of the feature set by
sorting the features by the explained variance.  It does this by
1) computing a covariance matrix for the features
2) finding the eigenvectors and eigenvalues of the matrix, principal components.
3) computing explained variance for the components and sorting them

Always standardize because PCA is sensitive to scaling


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
n_components = 10 # of of pca components to use for final accuracy

import numpy as np
import ocr_utils
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA


chars_to_train = range(48,58)
columnsXY=range(0,20)
column_str = 'column_sum{}'.format(list(columnsXY))

input_filters_dict = {'m_label': chars_to_train, 'font': 'E13B'}

# output  the character label and the image and column sums
output_feature_list = ['m_label','image',column_str] 

# read the complete image (20x20) = 400 pixels for each character
ds = ocr_utils.read_data(input_filters_dict=input_filters_dict, 
                            output_feature_list=output_feature_list, 
                            test_size=.2,
                            random_state=0)
windows_limit = 5000 # uses too much memory for my 32 bit windows computer so limit size of sample   
y_train = ds.train.features[0][:windows_limit]
X_train_image = ds.train.features[1][:windows_limit]
X_train = ds.train.features[2][:windows_limit]

y_test = ds.test.features[0]
X_test_image = ds.test.features[1]
X_test = ds.test.features[2]


cov_mat = np.cov(X_train_image.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals[:2*n_components])

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
var_exp = var_exp[:20]
cum_var_exp = cum_var_exp[:2*n_components]
title='explained variance'
plt.bar(range(1, len(var_exp)+1), var_exp, alpha=0.5, align='center',   label='individual explained variance')
plt.step(range(1, len(cum_var_exp)+1), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.title(title)
ocr_utils.show_figures(plt,title)

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(reverse=True)

# The eigenpairs with the highest explained variance
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w[:2*n_components,:])

X_train_pca = X_train_image.dot(w)
print ('projection of first dataset sample on first 2 eignvectors {}'.format(X_train_image[0].dot(w)))

markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan','orange','green','brown','lightblue','pink')

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], 
                X_train_pca[y_train==l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
title='features mapped to two principal components'
plt.title(title)
ocr_utils.show_figures(plt,title)

########################################################################################


pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_image)
X_test_pca = pca.transform(X_test_image)

lr = LogisticRegression(solver='liblinear',multi_class='auto')
logistic_fitted =lr.fit(X_train_pca, y_train)

print('\nPCA Train Accuracy: {:4.6f}, n_components={}'.format(accuracy_score(y_train, logistic_fitted.predict(X_train_pca)),pca.n_components))
print('PCA Test Accuracy: {:4.6f}, n_components={}'.format(accuracy_score(y_test, logistic_fitted.predict(X_test_pca)),pca.n_components))

title = 'train pc1 versus pc2'    
ocr_utils.plot_decision_regions(X=X_train_pca, y=y_train, classifier=lr, labels=['pc1','pc2'], title=title)

title = 'test pc1 versus pc2'  
ocr_utils.plot_decision_regions(X=X_test_pca, y=y_test, classifier=lr, labels=['pc1','pc2'], title=title)
X_train_pca = pca.fit_transform(X_train_image)
X_test_pca = pca.transform(X_test_image)

########################################################################################
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_image)
X_test_pca = pca.transform(X_test_image)

lr = LogisticRegression(solver='liblinear',multi_class='auto')
logistic_fitted = lr.fit(X_train_pca, y_train)

y_train_pred = logistic_fitted.predict(X_train_pca)
y_test_pred = logistic_fitted.predict(X_test_pca)

print('\nPCA Train Accuracy: {:4.6f}, n_components={}'.format(accuracy_score(y_train, y_train_pred),pca.n_components))
print('PCA Test Accuracy: {:4.6f}, n_components={}'.format(accuracy_score(y_test, y_test_pred),pca.n_components))

X_errors_image = X_test_image[y_test!=y_test_pred]
y_errors = y_test[y_test!=y_test_pred]
X_errors_pca = X_test_pca[y_test!=y_test_pred]

X_orig = X_train_image[:500]
title = 'originals'
X2D=np.reshape(X_orig, (X_orig.shape[0], ds.train.num_rows, ds.train.num_columns))
ocr_utils.montage(X2D,title=title)

X_orig = X_train_pca[:500]
title = 'inverse original'
X_inverse = pca.inverse_transform(X_orig)
X2D = np.reshape(X_inverse, (X_inverse.shape[0], ds.train.num_rows, ds.train.num_columns))
X2D = X2D - np.min(X2D)
ocr_utils.montage(X2D,title=title)

# change to a 2D shape 
X_errors2D=np.reshape(X_errors_image, (X_errors_image.shape[0], ds.train.num_rows, ds.train.num_columns))
ocr_utils.montage(X_errors2D,title='PCA Error Characters, components={}'.format (n_components))

title = 'inverse transform errors'
X_inverse = pca.inverse_transform(X_errors_pca)
X2D=np.reshape(X_inverse, (X_inverse.shape[0], ds.train.num_rows, ds.train.num_columns))
X2D = X2D - np.min(X2D)
ocr_utils.montage(X2D,title=title)

########################################################################################
kernel='rbf'  # really slow
pca = KernelPCA(n_components=2,kernel=kernel, gamma=15)

X_train_pca = pca.fit_transform(X_train_image)
X_test_pca = pca.transform(X_test_image)

lr = LogisticRegression(solver='liblinear',multi_class='auto')
logistic_fitted=lr.fit(X_train_pca, y_train)
y_train_pred = logistic_fitted.predict(X_train_pca)
y_test_pred = logistic_fitted.predict(X_test_pca)

print('\nKernel PCA Train Accuracy: {:4.6f}, n_components={}, kernel={}'.format(accuracy_score(y_train, y_train_pred), pca.n_components,kernel))
print('Kernel PCA Test Accuracy: {:4.6f}, n_components={}, kernel={}'.format(accuracy_score(y_test, y_test_pred),pca.n_components,kernel))

title = 'train kernel {} pc1 versus pc2'.format(kernel)    
ocr_utils.plot_decision_regions(X=X_train_pca, y=y_train, classifier=lr, labels=['pc1','pc2'], title=title)

title = 'test kernel {} pc1 versus pc2'.format(kernel)    
ocr_utils.plot_decision_regions(X=X_test_pca, y=y_test, classifier=lr, labels=['pc1','pc2'], title=title)




########################################################################################
# too slow on my computer

# pca = KernelPCA(n_components=n_components,kernel=kernel, gamma = 15)
# 
# X_train_pca = pca.fit_transform(X_train_image)
# X_test_pca = pca.transform(X_test_image)
# 
# print ('n_components={}'.format(pca.n_components))
# 
# lr = LogisticRegression()
# logistic_fitted = lr.fit(X_train_pca, y_train)
# 
# y_pred = logistic_fitted.predict(X_test_pca)
# print('\nKKernelPCA Train Accuracy: {:4.6f}, n_components={}, kernel={}'.format(accuracy_score(y_train, logistic_fitted.predict(X_train_pca)), pca.n_components, kernel))
# print('KernelPCA Test Accuracy: {:4.6f}, n_components={}, kernel={}'.format(accuracy_score(y_test, y_pred), pca.n_components, kernel))
# 
# X_errors_image = X_test_image[y_test!=y_pred]
# y_errors = y_test[y_test!=y_pred]
# 
# error_images = X_errors_image.shape[0]
# 
# # change to a 2D shape 
# X_errors2D=np.reshape(X_errors_image, (error_images, ds.train.num_rows, ds.train.num_columns))
# ocr_utils.montage(X_errors2D,title='Kernel {} KernelPCA Errors Character,components={}s'.format(kernel,n_components))

print ('\n########################### No Errors ####################################')

