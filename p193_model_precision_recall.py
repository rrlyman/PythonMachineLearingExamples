'''mode_precision_recall.py

    Precision and recall are measures of true positives. 
    Precision is also called positive predictive value. 
    Precision is "how useful the search results are".
    
    Recall is also called sensitivity
    Recall is  "how complete the results are".
    
    Combining them is the F1 score
    It is the harmonic meanof Precision and Recall
    
    Given a couple of E13B, compute the precision and recall values.
    Make a scorer using the F1 measure as the score and use
    grid search to find the parameters that give the highest F1 measure
    
    

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

import ocr_utils
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    y, X, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,51) , columns=(9,17), random_state=0) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    pipe_svc = Pipeline([('scl', StandardScaler()),
                ('clf', SVC(random_state=1))])

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    param_grid = [{'clf__C': param_range, 
                   'clf__kernel': ['linear']},
                     {'clf__C': param_range, 
                      'clf__gamma': param_range, 
                      'clf__kernel': ['rbf']}]
    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)

    pos_label=y_train[0]
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, pos_label=pos_label))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, pos_label=pos_label))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, pos_label=pos_label))

    scorer = make_scorer(f1_score, pos_label=pos_label)

    c_gamma_range = [0.01, 0.1, 1.0, 10.0]

    param_grid = [{'clf__C': c_gamma_range, 
                   'clf__kernel': ['linear']},
                     {'clf__C': c_gamma_range, 
                      'clf__gamma': c_gamma_range, 
                      'clf__kernel': ['rbf'],}]

    gs = GridSearchCV(estimator=pipe_svc, 
                                    param_grid=param_grid, 
                                    scoring=scorer, 
                                    cv=10,
                                    n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print('\nGrid Search f1 scoring best score: {}'.format(gs.best_score_))
    print('Grid Search f1 scoring best params: {}'.format(sorted(gs.best_params_.items())))
    print ('\n########################### No Errors ####################################')
