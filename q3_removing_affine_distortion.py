'''
Created on Jul 23, 2016

@author: richard
'''
'''
Created on Jul 12, 2016
This program shows how Principal Component Analysis removes affine 
transformation distortions.

Parallel lines in an image remain parallel after an affine transformation. 
For instance, if an image is rotated or sheared, lines remain parallel.

PCA and LDA can remove affine transformations.  This is shown by making 3 shapes
and then making a number of shear versions of the shapes.  Running 
Principal Component Analysis reduces the number of features necessary to
recognize the features during Logistic Regression with 100% accuracy, 
down to 2 from 400 (20 columns by 20 rows).

We make three images and then make about 80 copies of each image created by
shearing the original image.

Since there is very little noise introduced by the shearing, almost all of
the explained variance is due to the shearing. PCA finds eigenvectors
that line up with shearing.

1) For a couple of shapes, make sheared version.
2) train and print accuracies without PCA 
3) repeat, but use PCA first before training.
4) observe the improvement

Do the same thing for Linear Discriminant Analysis

@author: richard
'''

import numpy as np
import ocr_utils   
from sklearn.metrics import accuracy_score
white_space = 5

#########################################################################
# make a 3 basic images with about 80 sheared clones each

plus = np.zeros((20,20))
box = np.zeros((20,20))
vee = np.zeros((20,20))

plus[range(white_space, 20-white_space),9:10] = 1.0
plus[9:10,range(white_space, 20-white_space)] = 1.0

box[white_space,range(white_space, 20-white_space)] = 1.0 #top
box[20-white_space, range(white_space, 20 -white_space)] = 1.0 #bottom
box[range(white_space, 20-white_space), white_space] = 1.0 # left
box[range(white_space, 20-white_space), 20 - white_space] = 1.0  #right

for i in range(20):
    vee[i,19-int(i/2)] = 1.0
    vee[i,int(i/2)] = 1.0
    
# make some skewed versions of the shapes
import skimage.transform as tf

def shear(X, skew):
    rows = X.shape[0]
    cols = X.shape[1]    
    ratioY = skew*cols/rows
    matrix =  np.array( [[1, ratioY, 0] ,[0, 1, 0] ,[0, 0, 1 ]])                                         
    tp=tf.ProjectiveTransform(matrix=matrix) 
    f = tf.warp(X, tp)      
    return f

# make some skewed versions of the shapes
skewRange = np.linspace(-0.5,0.5,81)
images = np.empty((3*len(skewRange),20,20))
ys = np.empty((3*len(skewRange)))
# make sheared versions of shapes
for i,skew in enumerate(skewRange):
    images[3*i] = shear(plus,skew)
    images[3*i+1] = shear(box,skew)
    images[3*i+2] = shear(vee,skew)
    ys[3*i] = 0
    ys[3*i+1] = 1
    ys[3*i+2] = 2
    
title='skewed versions of shapes'
ocr_utils.montage(images,title=title) 

num_image=images.shape[0]
images_reshaped = np.reshape(images,(num_image, 20*20))

#########################################################################
# run a Logistic Regression on the raw features with 20 rows, 20 columns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train , X_test, y_train, y_test = train_test_split(images_reshaped, ys, test_size=0.3, random_state=0)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print('\nTrain Accuracy: {:4.6f} coefficients={}'.format(accuracy_score(y_train, y_train_pred), lr.coef_.shape))
print('Test Accuracy: {:4.6f} coefficients={}'.format(accuracy_score(y_test, y_test_pred), lr.coef_.shape))

#########################################################################
# run Principal Component analysis first, then Logistic Regression

from sklearn.decomposition import PCA
n_components = 2
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print('\nPCA components = {}'.format(pca.components_.shape))

lr = LogisticRegression()
logistic_fitted = lr.fit(X_train_pca, y_train)

y_train_pred = logistic_fitted.predict(X_train_pca)
y_test_pred = logistic_fitted.predict(X_test_pca)

print('\nPCA Train Accuracy: {:4.6f}, n_components={} coefficients={}'.format(accuracy_score(y_train, y_train_pred),pca.n_components,lr.coef_.shape))
print('PCA Test Accuracy: {:4.6f}, n_components={} coefficients={}'.format(accuracy_score(y_test, y_test_pred),pca.n_components,lr.coef_.shape))

X_errors_image = X_test[y_test!=y_test_pred]
y_errors = y_test[y_test!=y_test_pred]
X_errors_pca = X_test_pca[y_test!=y_test_pred]

# change to a 2D shape 
X_errors2D=np.reshape(X_errors_image, (X_errors_image.shape[0], 20, 20))
ocr_utils.montage(X_errors2D,title='PCA Error Images, components={}'.format (n_components))

X_combined = np.vstack((X_train_pca, X_test_pca))
y_combined = np.hstack((y_train, y_test))

ocr_utils.plot_decision_regions(
                                         X=X_combined,                                        
                                         y=y_combined,                                        
                                         classifier=lr,  
                                         labels = ['PC1','PC2']  ,     
                                         title='logistic_regression after 2 component PCA')

#########################################################################
# run Linear Discriminant Analysis first then Logistic Regression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
n_components = 2
lda = LDA(n_components=n_components)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
print('\nLDA components = {}'.format(pca.components_.shape))
lr = LogisticRegression()
logistic_fitted = lr.fit(X_train_lda, y_train)

y_train_pred = logistic_fitted.predict(X_train_lda)
y_test_pred = logistic_fitted.predict(X_test_lda)

print('\nLDA Train Accuracy: {:4.6f}, n_components={} coefficients={}'.format(accuracy_score(y_train, y_train_pred),lda.n_components,lr.coef_.shape))
print('LDA Test Accuracy: {:4.6f}, n_components={} coefficients={}'.format(accuracy_score(y_test, y_test_pred),lda.n_components,lr.coef_.shape))

X_errors_image = X_test[y_test!=y_test_pred]

# change to a 2D shape 
X_errors2D=np.reshape(X_errors_image, (X_errors_image.shape[0], 20, 20))
ocr_utils.montage(X_errors2D,title='LDA Error Images, components={}'.format (n_components))

X_combined = np.vstack((X_train_lda, X_test_lda))
y_combined = np.hstack((y_train, y_test))

ocr_utils.plot_decision_regions(
                                         X=X_combined,                                        
                                         y=y_combined,                                        
                                         classifier=lr,  
                                         labels = ['LDA1','LDA2']  ,     
                                         title='logistic_regression after 2 component LDA')

print ('\n########################### No Errors ####################################')
