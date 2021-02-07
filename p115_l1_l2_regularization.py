#!/usr/bin/python
''' l1_l2_regularization.py

    Show the effects of l1 versus l2 regulartion. 
    l1 introduces a weight penalty equal to the sum of the absolute weights
        times a given factor, lambda
        l1, tends to drive a number of weights to zero and thus yields a 
        sparse weight matrix
        
    l2 introduces a weight penalty equal to the sum of squares of the
        weights times lambda.
        l2, tends to reduces the size of the weights but does not drive
        them to 0
    

    1) get the data for all column sums in the e13b database
    2  run logistic regression both with l1 and l2 regulization printing
    out the accuracies and sampling of the coefficients.
    Show how the weights respond versus the regularization
     
    
    
Created on Jun 23, 2016

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

columnsXY = range(0,20)    
y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,49,50) , columns=columnsXY , test_size=0.3, nChars=1000, random_state=0) 

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', C=0.1, random_state=0, solver='liblinear',multi_class='auto')
lr.fit(X_train_std, y_train)
print('Training accuracy-l1 regularization:', lr.score(X_train_std, y_train))
print('Test accuracy-l1 regularization:', lr.score(X_test_std, y_test))
print('lr.intercept_ L1 regularization')
print('\t{}'.format(lr.intercept_))
print('lr.coef_ L1 regularization')
print('\t{}'.format(lr.coef_))


lr = LogisticRegression(penalty='l2', C=0.1, random_state=0, solver='liblinear',multi_class='auto')
lr.fit(X_train_std, y_train)
print('Training accuracy-l2 regularization:', lr.score(X_train_std, y_train))
print('Test accuracy-l2 regularization:', lr.score(X_test_std, y_test))
print('lr.intercept L2 regularization')
print('\t{}'.format(lr.intercept_))
print('lr.coef_ L2 regularization')
print('\t{}'.format(lr.coef_))

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 
         'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

def weight_graph(regularization = 'l1'):
    weights, params = [], []
    for c in np.arange(0, 6):
        lr = LogisticRegression(penalty=regularization, C=10**c, random_state=0, solver='liblinear',multi_class='auto')
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10**c)
    
    weights = np.array(weights)
    
    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:, column],
                 label=columnsXY[column+1],
                 color=color)
    
           
    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**(-5), 10**5])
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    title = 'regularization {}'.format(regularization)
    plt.title(title)
    plt.legend(loc='upper left')
    ax.legend(loc='upper center', 
              bbox_to_anchor=(1.38, 1.03),
              ncol=1, fancybox=True)
    ocr_utils.show_figures(plt,title + ' path')
    
weight_graph(regularization = 'l1')
weight_graph(regularization = 'l2')
print ('\n########################### No Errors ####################################')