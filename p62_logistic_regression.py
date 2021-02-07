#!/usr/bin/python
''' logitistic_function.py replaces the activation function by the 
        logistic function also known as the sigmoid function.
    The logistic function starts with the racetrack odds ratio p/(1-p)
    The logit function is the log of this
    Solving the inverse function for p yields the logit function
    The output predicts the probability of an input sample belonging to 
        a class label.
    This gives a estimate of the probability that a input set of features 
        belong to a target class.
    Works best when classes are linearly separable.
    For multiclasses when there are  more than 2 classes, uses One versus Rest.
    The effect of the regularization parameter in the logistic regression is
        shown.
    
    1) Plot the logistic (sigmoid) function
    2)Create the cost function to be minimized by using the negative of the
        log likelihood function.
    3) Plot the two curves that make up the cost function, one for a target y
        that equals 1 and one for a target y that equals 0
        Use the sklearn package to fit the input data to a single perceptron using 
        the logistic function for an activation function.
    4) Plot the decision regions for 3 target classes from the E13B training set
    5) Plot the weight coefficients using two different regulartion values     
    
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

y, X, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,49,50) , columns=(9,17),nChars=500) 

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
title='sigmoid'
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.title(title)
ocr_utils.show_figures(plt,title=title)

def cost_1(z):
    return - np.log(sigmoid(z))
                   
def cost_0(z):
    return - np.log(1 - sigmoid(z))

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y=1')

c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')
title='log cost'
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.title(title)
plt.tight_layout()
ocr_utils.show_figures(plt,title=title)

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
lr = LogisticRegression(C=1000.0, random_state=0, solver='lbfgs',multi_class='auto')
lr.fit(X_train_std, y_train)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
ocr_utils.plot_decision_regions(
                                         X=X_combined_std,                                        
                                         y=y_combined,                                        
                                         classifier=lr,         
                                         labels = labels, 
                                         test_idx=range(len(X_train_std),len(X_combined_std)),
                                         title='logistic_regression')


weights, params = [], []
for c in np.arange(0, 5):
    lr = LogisticRegression(C=10**c, random_state=0, solver='lbfgs',multi_class='auto')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[0])
    params.append(10**c)


title = 'regression_path'
weights, params = [], []
for c in np.arange(0, 5):
    lr = LogisticRegression(C=10**c, random_state=0, solver='lbfgs',multi_class='auto')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0], 
         label=labels[0])
plt.plot(params, weights[:, 1], linestyle='--', 
         label=labels[1])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.title(title)
ocr_utils.show_figures(plt,title=title)

print ('\n########################### No Errors ####################################')