'''k_fold_model_selection.py
k fold cross validation splits the training set into n parts and uses a 
different 1/n of the test set for each iteration.  It is good for
tuning  parameters as all samples are used, reducing the variance of the 
model performance.

Using a pipeline automates the steps by putting into a batch pipe
    1) scaling, 
    2) Principle Component Analysis, and
    3) training 
    
    StratifiedKFold returns lists of the indexes of the X samples and y
    target samples to be used for each fold
    
    The cross_val_score returns accuracy scores for each k-fold predictor
    from each fold.
    
Created on Jul 5, 2016

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


import ocr_utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis as LDA  

if __name__ == '__main__':
    #charsToTrain=range(48,58)
    chars_to_train = range(48,58)
    n_classes = len(chars_to_train)
    
    num_chars = 3000 #limit the number to speed up the calculation
    
    input_filters_dict = {'m_label': chars_to_train, 'font': 'E13B'}
    
    # output  the character label and the image and column sums
    output_feature_list = ['m_label','image'] 
    
    # read the complete image (20x20) = 400 pixels for each character
    ds = ocr_utils.read_data(input_filters_dict=input_filters_dict, 
                                output_feature_list=output_feature_list, 
                                random_state=0)
       
    y_train = ds.train.features[0][:num_chars]
    X_train = ds.train.features[1][:num_chars]
    
    # y_test = ds.test.features[0]-48
    # X_test = ds.test.features[1]
    # y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = charsToTrain , columns=range(0,20), nChars=1000, test_size=0.3,random_state=0)  
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    X_train , X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
    
    from sklearn.preprocessing import StandardScaler
    #  
    # sc = StandardScaler()
    # X_train_std = sc.fit_transform(X_train)
    # X_test_std = sc.fit_transform(X_test)
     
    # X_train, X_test, y_train, y_test = \
    #         train_test_split(X, y, test_size=0.20, random_state=1)
    
    from sklearn.decomposition import PCA
    
    from sklearn.pipeline import Pipeline
    
    num_planes = range(2,12)
    
    pca_scores =[]
    pca_std_dev =[]
    for num_PCA in num_planes:
        print ('number of Principal Components = {}'.format(num_PCA))
        pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=num_PCA, svd_solver='full')),
                    ('clf', LogisticRegression(random_state=1,multi_class='auto', solver='liblinear'))])

        pipe_lr.fit(X_train, y_train)
        print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
        
        kfold = StratifiedKFold(n_splits=10, random_state=1)
        
        scores = []
        for train_index, test_index in kfold.split(X_train, y_train):
            pipe_lr.fit(X_train[train_index], y_train[train_index])
            score = pipe_lr.score(X_train[test_index], y_train[test_index])
            scores.append(score)
            #print ('train {} samples: {}'.format(len(train), train))
            #print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train])[list(charsToTrain)], score))
            
        print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(estimator=pipe_lr, 
                                 X=X_train, 
                                 y=y_train, 
                                 cv=10,
                                 n_jobs=-1)
        print('CV accuracy scores: %s' % scores)
        print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
        pca_scores.append(np.mean(scores))
        pca_std_dev.append(np.std(scores))    
    
    plt.plot(num_planes, pca_scores,  marker='o')
    plt.ylabel('Accuracy')
    plt.xlabel('number of Principal Components')
    title = 'Accuracy versus number of Principal Components'
    plt.title(title)    
    plt.tight_layout()
    ocr_utils.show_figures(plt, title)
    
    plt.plot(num_planes, pca_std_dev,  marker='o')
    plt.ylabel('Standard Deviation')
    plt.xlabel('number of Principal Components')
    title = 'Standard Deviation versus number of Principal Components'
    plt.title(title)    
    plt.tight_layout()
    ocr_utils.show_figures(plt, title)
    
    pca_scores =[]
    pca_std_dev =[]
    for num_LDA in num_planes:
        print ('number of Principal Components = {}'.format(num_LDA))
        pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('lda', LDA(n_components=min(num_LDA,n_classes-1), solver='eigen')),
                    ('clf', LogisticRegression(random_state=1,multi_class='auto',solver='liblinear'))])

        kys = pipe_lr.get_params().keys()  
        print(kys)      
#         pipe_lr.set_params(lda__solver='eigen',clf__solver='liblinear',clf__multi_class='auto')             
        pipe_lr.fit(X_train, y_train)
        print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
        
        kfold = StratifiedKFold(n_splits=10, random_state=1)
        
        scores = []
        for train_index, test_index in kfold.split(X_train, y_train):
            pipe_lr.fit(X_train[train_index], y_train[train_index])
            score = pipe_lr.score(X_train[test_index], y_train[test_index])            
            scores.append(score)
            #print ('train {} samples: {}'.format(len(train), train))
            #print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train])[list(charsToTrain)], score))
            
        print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
        
        scores = cross_val_score(estimator=pipe_lr, 
                                 X=X_train, 
                                 y=y_train, 
                                 cv=10,
                                 n_jobs=-1)
        print('CV accuracy scores: %s' % scores)
        print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
        pca_scores.append(np.mean(scores))
        pca_std_dev.append(np.std(scores))    
    
    plt.plot(num_planes, pca_scores,  marker='o')
    plt.ylabel('Accuracy')
    plt.xlabel('number of Linear Discriminants')
    title = 'Accuracy versus number of Linear Discriminants'
    plt.title(title)    
    plt.tight_layout()
    ocr_utils.show_figures(plt, title)
    
    plt.plot(num_planes, pca_std_dev,  marker='o')
    plt.ylabel('Standard Deviation')
    plt.xlabel('number of Linear Discriminants')
    title = 'Standard Deviation versus number of Linear Discriminants'
    plt.title(title)    
    plt.tight_layout()
    ocr_utils.show_figures(plt, title)
    
    print ('\n########################### No Errors ####################################')