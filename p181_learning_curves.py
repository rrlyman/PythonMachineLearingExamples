'''
learning_curves.py

The learning curve shows the training and test accuracies versus the number
of training samples.  This can be used to determine if there are enough
training samples to sufficiently fit the training data.  If the curves
flatten out then there are enough samples.  If there is a big difference 
between the training and test accuracies then under and over fitting
can be determined.

The validation curve can be used to test a range of an estimation 
parameter such as the inverse regularization parameter, C, to see how the
test and accuracy varies.  This can be used to pick a value of C to reduce
over fitting and under fitting.

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
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import ocr_utils  
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    

    y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,51) , columns=(9,17), random_state=0) 
          
    pipe_lr = Pipeline([('scl', StandardScaler()),
                ('clf', LogisticRegression(penalty='l2', random_state=0, solver='lbfgs'))])

    train_sizes, train_scores, test_scores =\
                    learning_curve(estimator=pipe_lr, 
                    X=X_train, 
                    y=y_train, 
                    train_sizes=np.linspace(0.1, 1.0, 10), 
                    cv=10,
                    n_jobs=8)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, 
             color='blue', marker='o', 
             markersize=5, label='training accuracy')

    plt.fill_between(train_sizes, 
                     train_mean + train_std,
                     train_mean - train_std, 
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean, 
             color='green', linestyle='--', 
             marker='s', markersize=5, 
             label='validation accuracy')

    plt.fill_between(train_sizes, 
                     test_mean + test_std,
                     test_mean - test_std, 
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    title='learning_curve'
    plt.title(title)
    plt.tight_layout()
    ocr_utils.show_figures(plt,title)

    from sklearn.model_selection import validation_curve

    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(
                    estimator=pipe_lr, 
                    X=X_train, 
                    y=y_train, 
                    param_name='clf__C', 
                    param_range=param_range,
                    cv=10,
                    n_jobs=8)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean, 
             color='blue', marker='o', 
             markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(param_range, test_mean, 
             color='green', linestyle='--', 
             marker='s', markersize=5, 
             label='validation accuracy')

    plt.fill_between(param_range, 
                     test_mean + test_std,
                     test_mean - test_std, 
                     alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])
    title='validation_curve'
    plt.title(title)
    plt.tight_layout()
    ocr_utils.show_figures(plt,title)

    print ('\n########################### No Errors ####################################')        

