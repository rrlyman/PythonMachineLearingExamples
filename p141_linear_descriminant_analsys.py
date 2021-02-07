''' linear_descriminant_analsys.py 
Linear Descriminant Analysis seeks to find a reduced dimensionality of 
the features by maximizing class separability A hyperplane between features from 
each class is calculated that separates the classes.  A new sample can be classified 
depending on which side of the hyperplane it is on.

In a non linearly separable set of features, then the separation is not 
perfect, i.e. some samples from class A might end up on the class B side
of the hyperplane.



    Note:  
        1) the technique assumes that the data is normally distributed.
        This is a red flag.  Normally distributed data usually does not exist.
        It is a figment of the imagination.
        2) LDA is supervised, meaning the class labels are known for each
        input sample, whereas Principle Component Analysis is unsupervised, 
        needing no labels.
        
    Procedure:
        1) Standardize the training set, because LDA is sensitive to scaling
        2) For each class, compute the d -dimensional mean vector.
        3) Construct the between-class scatter matrix S B and the within-class 
        scatter matrix S w
        4) apply the steps of Principal Component Anaysis - eigenpairs,  etc
        
        
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
#from sklearn.lda import LDA
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis as LDA   
print_limit = 20
chars_to_train = range(48,58)
n_classes = len(chars_to_train)
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
   
y_train = ds.train.features[0]
X_train_image = ds.train.features[1]
X_train = ds.train.features[2]

y_test = ds.test.features[0]
X_test_image = ds.test.features[1]
X_test = ds.test.features[2]


from sklearn.preprocessing import StandardScaler
# 
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train_image)
X_test_std = sc.fit_transform(X_test_image)
# X_train_std = X_train_image
# X_test_std = X_test_image
unique_labels=np.unique(y_train)
num_unique_labels = len(unique_labels)
np.set_printoptions(precision=4)

mean_vecs = []

for label in unique_labels:
#for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
#     print('Mean Vector {}: {}\n'.format(label, mean_vecs[i]))

d = mean_vecs[0].shape[0] # number of features

S_W = np.zeros((d, d))
for label, mv in zip(unique_labels, mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row-mv).dot((row-mv).T)
    S_W += class_scatter
   
print('Within-class scatter matrix: {}x{}'.format(S_W.shape[0], S_W.shape[1]))

print('Class label distribution: %s'
    % np.bincount(np.array(y_train,dtype='int32'))[int(min(y_train)):])

d = S_W.shape[1] # number of features
S_W = np.zeros((d, d))
for label,mv in zip(unique_labels, mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

mean_overall = np.mean(X_train_std, axis=0)
#d = 13 # number of features
S_B = np.zeros((d, d))
for i,mean_vec in enumerate(mean_vecs):
    #n = X[y==i+1, :].shape[0]    
    n = X_train_std[y_train==np.unique(y_train)[i], :].shape[0]
    mean_vec = mean_vec.reshape(d, 1) # make column vector
    mean_overall = mean_overall.reshape(d, 1) # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))


# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs[:print_limit]:
    print(eigen_val[0])

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)][:20]
cum_discr = np.cumsum(discr)

plt.bar(range(1, len(discr)+1), discr, alpha=0.5, align='center',
        label='individual "discriminability"')
plt.step(range(1, len(cum_discr)+1), cum_discr, where='mid',
         label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
title='Discriminability'
plt.title(title)
plt.tight_layout()
ocr_utils.show_figures(plt,title)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
                      eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w[:print_limit])

X_train_lda = X_train_std.dot(w)

markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan','orange','green','brown','lightblue','pink')

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0], 
                X_train_lda[y_train==l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='upper right')
title='Projecting Feature Set onto New Feature Space'
plt.title(title)
plt.tight_layout()
ocr_utils.show_figures(plt,title)

###############################################################################3
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda =  lda.transform(X_test_std)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear', multi_class='auto')
lr = lr.fit(X_train_lda, y_train)

title = 'Linear Descriminant Analysis Training Set'
ocr_utils.plot_decision_regions(X_train_lda, y_train, classifier=lr, labels=['LD 1','LD 2'], title=title)

title = 'Linear Descriminant Analysis Test Set'

ocr_utils.plot_decision_regions(X_test_lda, y_test, classifier=lr, labels=['LD 1','LD 2'], title=title)


###############################################################################
n_components = 10
lda = LDA(n_components=min(n_components,n_classes-1))
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

print ('n_components={}'.format(lda.n_components))

lr = LogisticRegression(solver='liblinear', multi_class='auto')
logistic_fitted = lr.fit(X_train_lda, y_train)

from sklearn.metrics import accuracy_score

y_pred_train = logistic_fitted.predict(X_train_lda)
y_pred_test = logistic_fitted.predict(X_test_lda)
print('\nLDA Train Accuracy: {:4.6f}, n_components={}'.format(accuracy_score(y_train, y_pred_train), lda.n_components))
print('LDA Test Accuracy: {:4.6f}, n_components={}'.format(accuracy_score(y_test, y_pred_test), lda.n_components))

X_errors_image = X_test_image[y_test!=y_pred_test]
y_errors = y_test[y_test!=y_pred_test]

# change to a 2D shape 
X2D=np.reshape(X_errors_image, (X_errors_image.shape[0], ds.train.num_rows, ds.train.num_columns))
ocr_utils.montage(X2D,title='LDA E13B Error Character,components={}'.format(n_components))

###############################################################################
n_components = 10
lda = LDA(n_components=n_components-1, solver='eigen')
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

print ('n_components={}'.format(lda.n_components))

lr = LogisticRegression(solver='liblinear', multi_class='auto')
logistic_fitted = lr.fit(X_train_lda, y_train)

from sklearn.metrics import accuracy_score

y_pred_train = logistic_fitted.predict(X_train_lda)
y_pred_test = logistic_fitted.predict(X_test_lda)
print('\nLDA eigen Train Accuracy: {:4.6f}, n_components={}'.format(accuracy_score(y_train, y_pred_train), lda.n_components))
print('LDA eigen Test Accuracy: {:4.6f}, n_components={}'.format(accuracy_score(y_test, y_pred_test), lda.n_components))

X_errors_image = X_test_image[y_test!=y_pred_test]
y_errors = y_test[y_test!=y_pred_test]

# change to a 2D shape 
X2D=np.reshape(X_errors_image, (X_errors_image.shape[0], ds.train.num_rows, ds.train.num_columns))
ocr_utils.montage(X2D,title='LDA eigen E13B Error Character,components={}'.format(n_components))

print ('\n########################### No Errors ####################################')
