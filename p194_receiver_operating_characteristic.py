''' receiver_operating_characterist.py

A receiver operating characterist plot is a plot of the true positive rate
against the false positive rate for a dataset with binary outcomes

A threshold for determining whether a sample is positive or negative is
the independent variable that is varied to produce the values in the graph.

The AUC, Area Under the Curve can be calculated.  The closer to 1.0, the
better the classification.

Created on Jul 9, 2016

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
# from sklearn.metrics import make_scorer,roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
import ocr_utils  
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,precision_score,roc_curve, auc
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score    

if __name__ == '__main__':

    y, X, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,51) , columns=(9,17), random_state=0) 
    from sklearn.preprocessing import LabelEncoder

    # the ROC is for data with a binary outcome. Change the ASCII characters to 0,1
    le = LabelEncoder()
    y = le.fit_transform(y)
    le.transform((48,51))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    pipe_lr = Pipeline([('scl', StandardScaler()),
                        ('pca', PCA(n_components=2)),
                        ('clf', LogisticRegression(penalty='l2',random_state=0,C=100.0, solver='lbfgs'))])

    # X_train2 = X_train[:, [4, 14]]
    X_train2 = X_train
    
    kfold = StratifiedKFold(n_splits=3, random_state=1)
        
#         scores = []
#         for train_index, test_index in kfold.split(X_train, y_train):
#             pipe_lr.fit(X_train[train_index], y_train[train_index])
#             score = pipe_lr.score(X_train[test_index], y_train[test_index])            
#             scores.append(score)
    
    
    
    
    
    
    
#     cv = StratifiedKFold(y_train,n_folds=3,random_state=1)
    fig = plt.figure(figsize=(7, 5))

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    i=0
    for train_index, test_index in kfold.split(X_train, y_train):
        probas = pipe_lr.fit(X_train2[train_index], 
                             y_train[train_index]).predict_proba(X_train2[test_index])
        
        fpr, tpr, thresholds = roc_curve(y_train[test_index], 
                                         probas[:, 1], 
                                         pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        i=i+1
        plt.plot(fpr, 
                 tpr, 
                 lw=1, 
                 label='ROC fold %d (area = %0.2f)' 
                        % (i, roc_auc))

    plt.plot([0, 1], 
             [0, 1], 
             linestyle='--', 
             color=(0.6, 0.6, 0.6), 
             label='random guessing')

    mean_tpr /= kfold.get_n_splits(X_train)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 0, 1], 
             [0, 1, 1], 
             lw=2, 
             linestyle=':', 
             color='black', 
             label='perfect performance')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    title='Receiver Operator Characteristic'
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    ocr_utils.show_figures(plt,title)


    pipe_lr = pipe_lr.fit(X_train2, y_train)
    # y_pred2 = pipe_lr.predict(X_test[:, [4, 14]])
    y_pred2 = pipe_lr.predict(X_test)

    print('ROC AUC: %.3f' % roc_auc_score(y_true=y_test, y_score=y_pred2))
    print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred2))
    #===================================================================================================================================================
    #illustrates how to make a scorer using the precision evaluation value
    # for more than 2 classes for GridSearch
    # i.e. applies a binary scoring technique to multiclasses
    pos_label=range(48,58)
#     pre_scorer = make_scorer(score_func=precision_score, 
#                              pos_label=pos_label, 
#                              greater_is_better=True, 
#                              average='micro')

    from sklearn.svm import SVC
    y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = pos_label , nChars=4000, columns=(9,17), random_state=0) 
    pipe_svc = Pipeline([('scl', StandardScaler()),
                ('clf', SVC(random_state=1))])
    c_gamma_range = [0.01, 0.1, 1.0, 10.0]
     
    param_grid = [{'clf__C': c_gamma_range, 
                   'clf__kernel': ['linear']},
                     {'clf__C': c_gamma_range, 
                      'clf__gamma': c_gamma_range, 
                      'clf__kernel': ['rbf'],}]
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(estimator=pipe_svc, 
                                param_grid=param_grid, 
                                scoring='accuracy',
                                cv=5,
                                n_jobs=-1)


    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('\nSupport Vector Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    gs = gs.fit(X_train, y_train)
    print('Support Vector Machine Grid Search best score: {}'.format(gs.best_score_))
    print('Support Vector Machine Grid Search best params: {}\n'.format(gs.best_params_))

    print ('\n########################### No Errors ####################################')

