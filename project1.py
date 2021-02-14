#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 06:43:16 2020

@author: rstewart61
"""

from data import random_state, rebalance, limit_size, wifi, dexter, qsar_androgen, breast_cancer, synthetic1, synthetic2, half_moons, circles, phishing, titanic, spambase, bank, hypo_thyroid, mushrooms, abalone, polish_bankruptcy, online_news, occupancy, placement, pulsars, credit_card_default, pima

import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve, GridSearchCV, RandomizedSearchCV
#from sklearn.model_selection import StratifiedKFold

#from sklearn.linear_model import LassoCV
#from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import OrdinalEncoder
#from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from sklearn.metrics import roc_curve, roc_auc_score, get_scorer, auc, precision_score, recall_score, average_precision_score, balanced_accuracy_score, f1_score, precision_recall_curve, classification_report, confusion_matrix
from sklearn.metrics import plot_precision_recall_curve, plot_confusion_matrix

#from imblearn.over_sampling import SMOTE
from sklearn.base import clone

from imblearn.over_sampling import SMOTE

from scipy.stats import uniform, reciprocal
import os

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Fix with https://stackoverflow.com/a/63622906
# from skopt import BayesSearchCV

import warnings
warnings.filterwarnings('ignore', '.*Solver terminated early.*')
warnings.filterwarnings('ignore', '.*Maximum iterations.*')
warnings.filterwarnings('ignore', 'Stochastic Optimizer.*')
warnings.filterwarnings('ignore', '.*Liblinear failed to converge.*')
warnings.filterwarnings('ignore')

import json
import copy

from time import perf_counter
import math

# TODO: Save dexter to CSV or load from original data
# TODO: MLP Learning curve.
# TODO: Use PCA to reduce dimensions to see if KNN can have improved performance.

fig_num = 12
do_tuning = True
do_fine_tuning = False
size_limit = 10000
num_folds = 10
tuning_iterations = 200
tuning_granularity = 1000
do_probability = True
# https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
#scoring = 'accuracy'
#scoring = 'roc_auc'
#scoring = 'average_precision'
scoring = 'balanced_accuracy'

pd.set_option('display.max_columns', None)

def my_make_pipeline(estimator):
    return make_pipeline(#RobustScaler(),
                         StandardScaler(),
                         #PCA(n_components=3),
                         estimator)

"""
https://github.com/scikit-learn/scikit-learn/issues/5992
def pr_auc_score(y_true, y_score):
    precision, recall, thresholds = \
        precision_recall_curve(y_true, y_score[:, 1])
    return auc(recall, precision, reorder=True)

pr_auc_scorer = make_scorer(pr_auc_score, greater_is_better=True,
                            needs_proba=True)
"""

def skip(data_set_name, model_name):
    #if model_name not in ['Decision Tree', 'Ada Boost',  'MLP', 'SVM (Linear)', 'SVM (RBF)']:
    #if data_set_name == 'Polish Bankruptcy' and model_name == 'SVM (Linear)':
    #    return False
    return False

# Based on https://stackoverflow.com/a/50354131
def label_bars(text_format, **kwargs):
    ax = plt.gca()
    fig = plt.gcf()
    bars = ax.patches

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for bar in bars:
        text = text_format % (bar.get_width())
        color = 'white'
        va = 'center'
        ha = 'center'
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + bar.get_height() / 2
        if bar.get_window_extent(renderer).width < 10:
            color = 'black'
            ha = 'left'
            text_x = bar.get_x() + 1.5 * bar.get_width()

        ax.text(text_x, text_y, text, ha=ha, va=va, color=color, **kwargs)

def knn(n, p):
    estimator = KNeighborsClassifier(p=2, metric='minkowski', n_jobs=-1)
    # Avoid breaking learning_curve
    max_neighbors = int(math.sqrt(n))
    print('max_neighbors', max_neighbors)
    param_grid = [{
            # Only odd numbers for K
            'kneighborsclassifier__n_neighbors':
                [2 * x + 1 for x in range(int(max_neighbors/2-1))],
                #np.linspace(start=1, stop=31, num=30).astype(int),
            'kneighborsclassifier__p': np.linspace(start=1, stop=8, num=tuning_granularity),
            'kneighborsclassifier__weights': ['uniform', 'distance']
            }]
    return estimator, param_grid

def svm_linear(n, p):
    """
    estimator = CalibratedClassifierCV(LinearSVC(random_state=random_state, max_iter=10000), cv=num_folds)
    param_grid = {'calibratedclassifiercv__base_estimator__max_iter': np.linspace(start=100, stop=3000, num=tuning_granularity).astype(int),
                  'calibratedclassifiercv__base_estimator__C': np.logspace(-4, 3, num=tuning_granularity)}
    """
    max_iter = int(n*1.5)
    estimator = SVC(kernel='linear', random_state=random_state, max_iter=20000,
                    probability=do_probability)
    param_grid = [{'svc__max_iter': np.linspace(start=1, stop=max_iter, num=max_iter).astype(int),
                  'svc__C': np.logspace(-5, 3, num=tuning_granularity)}]
    return estimator, param_grid

def svm_poly(n, p):
    estimator = SVC(kernel='poly', random_state=random_state, max_iter=20000,
                    probability=do_probability)
    param_grid = {'svc__gamma': np.logspace(start=3, stop=-12, num=tuning_granularity),
                  'svc__max_iter': np.linspace(start=1, stop=n, num=n).astype(int),
                  'svc__C': np.logspace(-5, 5, num=tuning_granularity),
                  'svc__degree': np.linspace(start=0, stop=20, num=tuning_granularity).astype(int)}
    return estimator, param_grid

def svm_rbf(n, p):
    if n < 500:
        max_iter_plot = 200
        start_gamma = -13
        end_gamma = -4
        end_C = 8
    else:
        max_iter_plot = 2500
        start_gamma = -2
        end_gamma = 5
        end_C = 3
    estimator = SVC(kernel='rbf', random_state=random_state, max_iter=20000,
                    probability=do_probability)
    param_grid = [{'svc__gamma': np.logspace(start=start_gamma, stop=end_gamma, num=tuning_granularity),
                  'svc__max_iter': np.linspace(start=1, stop=max_iter_plot, num=tuning_granularity).astype(int),
                  'svc__C': np.logspace(-2, end_C, num=tuning_granularity)},
                  {'svc__gamma': np.logspace(start=start_gamma, stop=end_gamma, num=tuning_granularity),
                  'svc__max_iter': np.linspace(start=1, stop=n, num=n).astype(int),
                  'svc__C': np.logspace(-5, 5, num=tuning_granularity),
                  'svc__degree': np.linspace(start=0, stop=20, num=tuning_granularity).astype(int)}]
    return estimator, param_grid

def decision_tree(n, p):
    if n > 500:
        max_leaf_nodes = 250
    else:
        max_leaf_nodes = 25
    estimator = DecisionTreeClassifier(random_state=random_state)
    param_grid = [{#'decisiontreeclassifier__criterion': ['gini','entropy'],
                  #'decisiontreeclassifier__min_impurity_decrease': np.logspace(start=-0.5, stop=-10, num=1000),
                  'decisiontreeclassifier__max_depth': np.linspace(start=1, stop=20, num=tuning_granularity).astype(int),
                  'decisiontreeclassifier__max_leaf_nodes': np.linspace(start=2, stop=max_leaf_nodes, num=tuning_granularity).astype(int), #np.linspace(start=1, stop=51, num=50).astype(int)
                  #'decisiontreeclassifier__min_samples_leaf': np.linspace(start=0.001, stop=0.1, num=tuning_granularity), #np.linspace(start=1, stop=51, num=50).astype(int)
                   },
                  {'decisiontreeclassifier__ccp_alpha': np.logspace(start=-10, stop=5, num=tuning_granularity)},
                  {'decisiontreeclassifier__max_features': np.linspace(start=1, stop=p, num=p).astype(int)},
                  {'decisiontreeclassifier__min_impurity_decrease':  np.logspace(start=-0.5, stop=-10, num=1000)},
                  {'decisiontreeclassifier__min_samples_split': np.linspace(start=2, stop=30, num=30).astype(int)},
                  {'decisiontreeclassifier__min_samples_leaf': np.linspace(start=1, stop=30, num=30).astype(int)}]
    return estimator, param_grid

def perceptron(n, p):
    """
    param_grid = {
            'calibratedclassifiercv__base_estimator__alpha': np.logspace(start=-1, stop=-5, num=tuning_granularity),
            'calibratedclassifiercv__base_estimator__eta0': np.logspace(start=-1, stop=-5, num=tuning_granularity),
            'calibratedclassifiercv__base_estimator__max_iter': np.linspace(start=100, stop=2001, num=tuning_granularity).astype(int),
            'calibratedclassifiercv__base_estimator__penalty': ['l2','l1','elasticnet', None]
        }
    """
    #estimator = CalibratedClassifierCV(Perceptron(random_state=random_state), cv=num_folds)
    #estimator = SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None, random_state=random_state)
    param_grid = {
            'perceptron__alpha': np.logspace(start=-1, stop=-5, num=tuning_granularity),
            'perceptron__eta0': np.logspace(start=-1, stop=-5, num=tuning_granularity),
            'perceptron__max_iter': np.linspace(start=1, stop=50, num=tuning_granularity).astype(int),
            'perceptron__penalty': ['l2','l1','elasticnet', None]
        }
    estimator = Perceptron(random_state=random_state)
    return estimator, param_grid

def mlp(n, p):
    max_iter = 15
    if n > 500:
        max_iter = 400
    # nodes per layer
    param_grid = [{
            'mlpclassifier__hidden_layer_sizes': list(zip(
                    np.linspace(start=1, stop=min(n, p), num=min(n, p)).astype(int)
                    )),
            #'mlpclassifier__alpha': np.logspace(start=-1, stop=-8, num=tuning_granularity),
            'mlpclassifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
            #'mlpclassifier__tol': np.logspace(start=-2, stop=-10, num=tuning_granularity),
            'mlpclassifier__learning_rate_init': np.logspace(start=5, stop=-7, num=tuning_granularity),
            'mlpclassifier__max_iter': np.linspace(start=1, stop=max_iter, num=tuning_granularity).astype(int)
        },
        {'mlpclassifier__learning_rate': ['invscaling'],
         'mlpclassifier__power_t': np.linspace(start=0.1, stop=2.0, num=tuning_granularity)},
        {'mlpclassifier__alpha': np.logspace(start=-1, stop=-8, num=tuning_granularity)},
        {            'mlpclassifier__hidden_layer_sizes': list(zip(
                    np.linspace(start=1, stop=min(n, p), num=min(n, p)).astype(int),
                    np.linspace(start=1, stop=min(n, p), num=min(n, p)).astype(int)
                    )),
        }
         ]
    # solver, alpha, power_t with learning_rate='invscaling', momentum with nesterovs_momentum, tol with n_iter_no_change
    # MLP very slow without early stopping
    estimator = MLPClassifier(early_stopping=False, solver='sgd', random_state=random_state)
    return estimator, param_grid


def ada_boost(n, p):
    estimator = GradientBoostingClassifier(random_state=random_state)
    param_grid = [{
                    "gradientboostingclassifier__learning_rate": np.logspace(start=-3, stop=1, num=tuning_granularity),
                    "gradientboostingclassifier__max_depth": np.linspace(start=1, stop=10, num=10).astype(int),
                    #"gradientboostingclassifier__max_features": np.linspace(start=1, stop=31, num=30).astype(int),
                    #"gradientboostingclassifier__min_samples_leaf": np.linspace(start=0.0001, stop=0.1, num=tuning_granularity),
                    "gradientboostingclassifier__max_leaf_nodes": np.linspace(start=2, stop=40, num=tuning_granularity).astype(int),
                    "gradientboostingclassifier__n_estimators": np.linspace(start=1, stop=min(n, p*5), num=tuning_granularity).astype(int)
                    #"gradientboostingclassifier__subsample": np.linspace(start=0.1, stop=1.0, num=tuning_granularity)
                },
                  {'gradientboostingclassifier__ccp_alpha': np.logspace(start=-10, stop=5, num=tuning_granularity)},
                  {'gradientboostingclassifier__max_features': np.linspace(start=1, stop=p, num=p).astype(int)},
                  {'gradientboostingclassifier__min_impurity_decrease':  np.logspace(start=-0.5, stop=-10, num=1000)},
                  {'gradientboostingclassifier__min_samples_split': np.linspace(start=2, stop=30, num=30).astype(int)},
                  {'gradientboostingclassifier__min_samples_leaf': np.linspace(start=1, stop=30, num=30).astype(int)},
                  {'gradientboostingclassifier__learning_rate': np.logspace(start=-4, stop=1, num=tuning_granularity)},
                  {'gradientboostingclassifier__subsample': np.linspace(start=0.1, stop=1.0, num=tuning_granularity)}
                  ]
    return estimator, param_grid

def model_label(f):
    label_map = {
            decision_tree: 'Decision Tree',
            knn: 'KNN',
            ada_boost: 'Ada Boost',
            svm_linear: 'SVM (Linear)',
            svm_poly: 'SVM (Poly)',
            svm_rbf: 'SVM (RBF)',
            mlp: 'MLP', #'Neural Net',
            perceptron: 'Perceptron'
            }
    return label_map[f]

def label(d):
    if callable(d):
        d = d.__name__
    return d.replace('_', ' ').title()

def get_path(title, data_set_name, model_name=None):
    path = 'plots/' + scoring + '/' + data_set_name + '/'
    if model_name is not None:
        path += model_name + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '_' + title

def save_fig(title, data_set_name, model_name=None, width=6, height=3):
    fig = plt.gcf()
    fig.set_size_inches(width, height)
    plt.savefig(get_path(title, data_set_name, model_name) + '.png', bbox_inches='tight')
    plt.close()

# https://stackoverflow.com/a/7100163
def save_json(o, title, data_set_name, model_name=None):
    with open(get_path(title, data_set_name, model_name) + '.json', 'w') as outfile:
        json.dump(o, outfile)

# https://stackoverflow.com/a/7100163
def load_json(title, data_set_name, model_name=None):
    filename = get_path(title, data_set_name, model_name) + '.json'
    print('Opening file %s' % filename)
    if not os.path.exists(filename):
        return None
    with open(filename) as infile:
        return json.load(infile)
    return None

#############
# ROC AUC
#############
def plot_roc_curve(title, fig_num, y_test, y_predict):
    fpr, tpr, threshold = roc_curve(y_test, y_predict, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(fig_num)
    plt.plot(fpr, tpr, label='%s (area = %0.3f)' % (title, roc_auc))

####################
# Precision Recall
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/
####################
def my_plot_precision_recall_curve(title, fig_num, y_test, y_predict):
    precision, recall, threshold = precision_recall_curve(y_test, y_predict, pos_label=1)
    #pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_test, y_predict)
    plt.figure(fig_num)
    plt.plot(recall, precision, label='PR for %s (average = %0.3f)' % (model_label(model), avg_precision))

def plot_learning_curve(type_name, data_set_name, model_name, pipe, X_train, y_train, plot_times):
    if skip(data_set_name, model_name):
        return
    global fig_num
    ###################################################
    # learning curve, from Machine Learning in Python
    ###################################################
    print()
    print('Plotting %s learning curve for %s, %s' % (type_name, data_set_name, model_name))
    print()
    training_start = 0.1
    if model_name == 'KNN':
        training_start = pipe.get_params()['kneighborsclassifier__n_neighbors'] * num_folds / X_train.shape[0]
        print('KNN training start hack', training_start, training_start * X_train.shape[0])
    train_sizes, train_scores, test_scores, train_times, query_times =\
            learning_curve(estimator=clone(pipe),
            X=X_train,
            y=y_train,
            train_sizes=np.linspace(training_start, 1.0, 10),
            cv=num_folds,
            n_jobs=-1,
            random_state=random_state,
            shuffle=True,
            scoring=scoring,
            verbose=1,
            return_times=True
            )
    if plot_times:
        plt.figure(11)
        plt.plot(train_sizes, np.mean(train_times, axis=1), label=model_label(model))
        plt.figure(12)
        plt.plot(train_sizes, np.mean(query_times, axis=1), label=model_label(model))

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    fig_num += 1
    plt.figure(fig_num)
    plt.title('%s learning curve for %s/%s' % (type_name, data_set_name, model_label(model)))
    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='Training score')
    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='Learning score')
    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training examples')
    plt.ylabel(label(scoring))
    """
    if (max(train_mean) < 0.75):
        plt.legend(loc='upper left')
    else:
        plt.legend(loc='lower right')
    """
    plt.legend()
    plt.ylim([0.0, 1.03])
    #plt.savefig('plots/' + title + '_learning_curve.png', bbox_inches='tight')
    save_fig('%s_learning_curve' % (type_name), data_set_name, model_name)

@ignore_warnings(category=ConvergenceWarning)
def tune_hyperparameters(data_set_name, model_name, pipe, param_grid, X_train, y_train):
    ##########################
    # Hyperparameter tuning
    ##########################
    #X_train, y_train = limit_size(data_set_name, X_train, y_train, 1000)
    print()
    param_grid = copy.copy(param_grid)
    to_remove = None
    for k in param_grid.keys():
        if k.endswith('__max_iter'):
            to_remove = k
    if to_remove is not None:
        del param_grid[k]

    cached_params = load_json('best_params', data_set_name, model_name)
    if cached_params is not None:
        print('Reusing cached params: ', cached_params)
        best_model = clone(pipe)
        best_model.set_params(**cached_params)
        best_model.fit(X_train, y_train)
    else:
        print('Starting hyperparameter tuning for %s/%s' % (data_set_name, model_name))
        """
        clf = BayesSearchCV(pipe, param_grid, cv=num_folds,
                            scoring=scoring, n_jobs=-1, n_iter=tuning_iterations,
                            random_state=random_state, verbose=1)
        clf.fit(X_train, y_train)
        return clf.best_estimator_
        """

        #clf = GridSearchCV(pipe, param_grid, cv=num_folds, scoring=scoring, n_jobs=-1, verbose=2)
        clf = RandomizedSearchCV(pipe, param_grid, cv=num_folds,
                                 scoring=scoring, n_jobs=-1, n_iter=tuning_iterations,
                                 random_state=random_state, verbose=1)
        clf.fit(X_train, y_train)

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        best_model = clf.best_estimator_

    current = best_model

    if skip(data_set_name, model_name):
        return current

    if do_fine_tuning:
        num_data_points = 20
        param_grid = copy.copy(param_grid)
        for iterations in range(3):
            num_changes = 0
            for param in param_grid.keys():
                param_range = param_grid[param]
                if (len(param_range) > num_data_points):
                    every = int(len(param_range) / num_data_points)
                    param_range = param_range[::every]
                    param_range.sort()
                if np.issubdtype(type(param_range[0]), int):
                    param_range = np.unique(param_range)
                print('Fine tuning %s for %s/%s in range: %s' % (param, data_set_name, model_name, param_range))
                print()
                next_clf = GridSearchCV(current, {param: param_range}, cv=num_folds,
                                        scoring=scoring, n_jobs=-1, verbose=1)
                next_clf.fit(X_train, y_train)
                prev_value = current.get_params()[param]
                current = next_clf.best_estimator_
                curr_value = current.get_params()[param]
                """
                if np.issubdtype(type(curr_value), float):
                    index = np.argwhere(np.isclose(param_range, curr_value))
                elif param == 'mlpclassifier__hidden_layer_sizes':
                    for i in range(len(param_range)):
                        if param_range[i][0] == curr_value[0]:
                            index = [[i]]
                else:
                    index = np.argwhere(param_range == curr_value)
                if len(index) > 0:
                    index = index[0][0]
                    if index > 0 and index < len(param_range)-1:
                        if np.issubdtype(type(curr_value), float):
                            param_grid[param] = np.linspace(start=param_range[index-1], stop=param_range[index+1], num=num_data_points)
                        if np.issubdtype(type(curr_value), int):
                            start = param_range[index-1]
                            stop = param_range[index+1]
                            if start != stop:
                                num = min(stop - start + 1, num_data_points)
                                param_grid[param] = np.linspace(start=start, stop=stop, num=num).astype(int)
                        if param == 'mlpclassifier__hidden_layer_sizes':
                            print('Special tuning logic for mlpclassifier__hidden_layer_sizes')
                            start = param_range[index-1][0]
                            stop = param_range[index+1][0]
                            if start != stop:
                                num = min(stop - start + 1, num_data_points)
                                param_grid[param] = list(zip(
                                        np.linspace(start=start, stop=stop, num=num).astype(int)
                                        ))
                else:
                    print('Could not find %s in %s' % (curr_value, param_range))
                """
                print()
                if prev_value != curr_value:
                    num_changes += 1
                    print('Updated %s from %s to %s' % (param, prev_value, curr_value))
                else:
                    print('%s unchanged' % (param))
            if num_changes == 0:
                break


        best_params = {}
        for param in param_grid.keys():
            best_value = current.get_params()[param]
            if np.issubdtype(type(best_value), int):
                best_value = int(best_value)
            elif np.issubdtype(type(best_value), float):
                best_value = float(best_value)
            elif param == 'mlpclassifier__hidden_layer_sizes':
                print('Special serialization logic for mlpclassifier__hidden_layer_sizes:', best_value)
                best_value = (int(best_value[0]),)
            best_params[param] = best_value
        print(best_params)
        save_json(best_params, 'best_params', data_set_name, model_name)
    return current

def is_log(seq):
    if not isinstance(seq[0], np.float64):
        return False
    if len(seq) < 3:
        return False
    d1 = np.abs(seq[0] - seq[1])
    d2 = np.abs(seq[-1] - seq[-2])
    #print(d1, d2, np.abs(d2 - d1), max(d1, d2), np.abs(d2 - d1) / max(d1, d2))
    if np.abs(d2 - d1) / max(d1, d2) < 0.5:
        return False
    return True

#####################
# Validation curve
# From Machine Learning with Python
#####################
@ignore_warnings(category=ConvergenceWarning)
def plot_validation_curves(data_set_name, model_name, pipe, param_grid, X_train, y_train):
    if skip(data_set_name, model_name):
        return

    global fig_num
    num_data_points = 20
    best_params = pipe.get_params()
    for param, param_range in param_grid.items():
        if (len(param_range) > num_data_points):
            every = int(len(param_range) / num_data_points)
            param_range = param_range[::every]
            param_range.sort()
        best_param_value = None
        if param in best_params:
            best_param_value = best_params[param]
        print()
        print('Plotting validation curve for %s, %s, %s' % (data_set_name, model_name, param))
        train_scores, test_scores = validation_curve(
            estimator=clone(pipe),
            X=X_train,
            y=y_train,
            param_name=param,
            param_range=param_range,
            scoring=scoring,
            cv=num_folds,
            n_jobs=-1,
            verbose=1)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        fig_num += 1
        plt.figure(fig_num)
        plot_range = param_range
        scale = 'linear'
        if is_log(param_range):
            scale = 'log'
            plt.xscale('log')
        param_name = param.split('__')[-1]

        best_param_value_str = str(best_param_value)
        if isinstance(best_param_value, np.float64):
            if scale == 'log':
                best_param_value_str = '%.3e' % (best_param_value)
            else:
                best_param_value_str = '%.3f' % (best_param_value)

        if isinstance(param_range[0], str):
            for i in range(len(param_range)):
                if param_range[i] is None:
                    param_range[i] = 'None'
            """
            plt.bar(x=plot_range, height=train_mean, yerr=train_std)
            plt.title('Validation for %s/%s %s' % (data_set_name, model_name, param_name))
            plt.title('Hyperparameter tuning improvements for %s' % (data_set_name))
            plt.xlabel('%s [best=%s]' % (param_name, best_param_value_str))
            plt.ylabel(scoring)
            save_fig('validation_curve_' + param_name, data_set_name, model_name)
            print('Created validation bar chart for ', param, '=', best_param_value)
            continue
            """
        elif param_name == 'hidden_layer_sizes':
            # best_param_value = len(best_param_value)
            # plot_range = [len(x) for x in param_range]
            if best_param_value:
                best_param_value = best_param_value[0]
            plot_range = [x[0] for x in param_range]

        is_convergence = param_name in ['max_iter', 'n_estimators']
        c = 0
        if is_convergence:
            #c = max(test_mean)
            c = test_mean[-1]
            plt.axhline(y=c, linestyle='--')
            plt.title('Convergence Plot for %s/%s' % (data_set_name, model_name))
        else:
            plt.title('Model Complexity Curve for %s/%s %s' % (data_set_name, model_name, label(param_name)))
        plt.plot(plot_range, train_mean,
                 color='blue', marker='o',
                 markersize=5, label='Training ' + label(scoring))
        plt.fill_between(plot_range, train_mean + train_std,
                         train_mean - train_std, alpha=0.15,
                         color='blue')
        plt.plot(plot_range, test_mean,
                 color='green', linestyle='--',
                 marker='s', markersize=5,
                 label='Validation ' + label(scoring))
        plt.fill_between(plot_range, test_mean + test_std,
                         test_mean - test_std, alpha=0.15,
                         color='green')
        if is_convergence:
            #plt.xlabel('Number of Iterations [convergence at %s]' % (best_param_value_str))
            if param_name == 'max_iter':
                plt.xlabel('Number of Iterations')
            elif param_name == 'n_estimators':
                plt.xlabel('Number of Estimators')
            plt.ylabel('%s [convergence at %.2f]' % (label(scoring), c))
        else:
            plt.xlabel('%s [best=%s]' % (label(param_name), best_param_value_str))
            if best_param_value is not None:
                plt.axvline(x=best_param_value, linestyle='--')
            plt.ylabel(label(scoring))
        plt.legend()
        save_fig('validation_curve_' + param_name, data_set_name, model_name)
        print('Created validation curve for ', param, '=', best_param_value)
        print()

def my_plot_confusion_matrix(title, data_set_name, model_name, estimator, X_test, y_test):
    global fig_num
    fig_num += 1
    disp = plot_confusion_matrix(estimator, X_test, y_test, normalize='all')
    disp.ax_.set_title('%s confusion matrix for %s %s' % (title, data_set_name, model_name))
    """
    # From Python Machine Learning 3rd Edition
    confmat = confusion_matrix(y_true=y_test, y_pred=y_predict)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    plt.title('%s confusion matrix for %s %s' % (title, data_set_name, model_name))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j+0.1, y=i+0.1,
                    s=confmat[i, j],
                    va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    """
    save_fig('confusion_matrix_%s' % (title), data_set_name, model_name)

def summarize(d):
    m = np.round(np.mean(d), 4)
    std = np.round(np.std(d), 4)
    return (m, std)

def score_model(scoring, y_true, y_predict, pos_label=1):
    """
    print('%s: y_true=%s, y_predict=%s, sensitivity=%s, specificity=%s, balanced_accuracy=%s' %
          (scoring, np.bincount(y_true), np.bincount(y_predict),
          recall_score(y_true, y_predict),
          recall_score(y_true, y_predict, pos_label=0),
          balanced_accuracy_score(y_true, y_predict)))
    """

    if scoring == 'average_precision':
        return average_precision_score(y_true, y_predict, pos_label=pos_label)
    if scoring == 'balanced_accuracy':
        return balanced_accuracy_score(y_true, y_predict)
    if scoring == 'f1':
        return f1_score(y_true, y_predict)
    if scoring == 'roc_auc':
        return roc_auc_score(y_true, y_predict)
    if scoring == 'precision':
        return precision_score(y_true, y_predict)
    if scoring == 'recall' or scoring == 'sensitivity':
        return recall_score(y_true, y_predict)
    if scoring == 'specificity':
        # https://stackoverflow.com/a/59396145
        return recall_score(y_true, y_predict, pos_label=0)

def score_train_and_test(estimator, scoring,
                         X_train, y_train, X_test, y_test, fig_offset, model):
    if hasattr(estimator, 'best_estimator_'):
        estimator = estimator.best_estimator_
    probas_test = estimator.predict_proba(X_test)
    y_test_predict = estimator.predict(X_test)

    #probas_train = estimator.predict_proba(X_train)
    y_train_predict = estimator.predict(X_train)

    #orig_scores = cross_val_score(pipe, X_test, y_test, scoring=scoring, cv=num_folds, n_jobs=-1)
    train_score = score_model(scoring, y_train, y_train_predict)
    test_score = score_model(scoring, y_test, y_test_predict)
    if scoring == 'balanced_accuracy':
        plot_roc_curve(model_label(model), fig_offset+1, y_test, probas_test[:, 1])
    #plot_precision_recall_curve(model_label(model), fig_offset+2, y_test, y_test_predict)

    return np.round(train_score, 3), np.round(test_score, 3)

def evaluate_model(data_set_name, model, X_train, y_train, X_test, y_test):
    global fig_num
    (n, p) = X_train.shape
    estimator, param_grids = model(n, p)
    pipe = my_make_pipeline(estimator)

    # Before hyperparameter tuning
    start_learning = perf_counter()
    result = pipe.fit(X_train, y_train)
    learning_time = perf_counter() - start_learning

    orig_train_score, orig_test_score = score_train_and_test(result, scoring, X_train, y_train, X_test, y_test, 0, model)
    """
    #orig_scores = cross_val_score(pipe, X_test, y_test, scoring=scoring, cv=num_folds, n_jobs=-1)
    orig_score = score_model(y_test, y_predict)
    plot_precision_recall_curve(model_label(model), 2, y_test, y_predict)
    probas = result.predict_proba(X_test)
    plot_roc_curve(model_label(model), 1, y_test, probas[:, 1])
    """

    start_query = perf_counter()
    pipe.predict(X_train)
    #pipe.predict(X_test)
    query_time = perf_counter() - start_query
    my_plot_confusion_matrix('Initial train', data_set_name, model_label(model), result, X_train, y_train)
    my_plot_confusion_matrix('Initial test', data_set_name, model_label(model), result, X_test, y_test)

    plot_learning_curve('Initial', data_set_name, model_label(model), pipe, X_train, y_train, True)
    if do_tuning:
        first = True
        for param_grid in param_grids:
            if first:
                tuned_estimator = tune_hyperparameters(data_set_name, model_label(model), pipe, param_grid, X_train, y_train)
                plot_learning_curve('Tuned', data_set_name, model_label(model), tuned_estimator, X_train, y_train, False)
                my_plot_confusion_matrix('Tuned train', data_set_name, model_label(model), tuned_estimator, X_train, y_train)
                my_plot_confusion_matrix('Tuned test', data_set_name, model_label(model), tuned_estimator, X_test, y_test)
                tuned_train_score, tuned_test_score = score_train_and_test(tuned_estimator, scoring, X_train, y_train, X_test, y_test, 2, model)
                #probas = tuned_estimator.predict_proba(X_test)
                #plot_roc_curve(model_label(model), 3, y_test, probas[:, 1])
                plot_validation_curves(data_set_name, model_label(model), tuned_estimator, param_grid, X_train, y_train)
                first = False
            else:
                pass
                # Don't use tuned_estimator here
                # plot_validation_curves(data_set_name, model_label(model), pipe, param_grid, X_train, y_train)

    else:
        tuned_train_score, tuned_test_score = orig_train_score, orig_test_score
        tuned_estimator = estimator

    return result, tuned_estimator, orig_train_score, orig_test_score, tuned_train_score, tuned_test_score, np.round(learning_time, 4), np.round(query_time, 4)
    #return summarize(orig_scores), summarize(tuned_scores), np.round(learning_time, 4), np.round(query_time, 4)
    # on older version of scikit learn
    #disp = plot_precision_recall_curve(pipe, X_test, y_test)
    #disp.ax_.set_title('2-class Precision-Recall curve: '
    #                   'AP={0:0.2f}'.format(pr_auc))

    #scores = cross_val_score(estimator=pipe, X=X_train, y=y_train, cv=num_folds, n_jobs=-1)
    #accuracy = np.mean(scores)
    #print('%s Test Accuracy: %.3f, ROC AUC: %.3f, PR AUC: %.3f, avg precision score: %.3f'
    #      % (model_label(model), accuracy, roc_auc, pr_auc, avg_precision))

# Based on https://matplotlib.org/3.3.1/gallery/lines_bars_and_markers/bar_stacked.html
def comparison_bar_chart_orig(data_set_name, scoring, labels, legend_labels, train, test, text_format='%.2f', **kwargs):
    width = 0.75
    fig, ax = plt.subplots()
    ax.bar(labels, test, width, label=legend_labels[0])
    diffs = []
    for i in range(len(train)):
        diffs.append(train[i] - test[i])
    ax.bar(labels, diffs, width, bottom=test, label=legend_labels[1])
    ax.set_ylabel(label(scoring))
    ax.set_title('%s for %s' % (label(scoring), data_set_name))
    ax.legend(loc='lower left')

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for bar in ax.patches:
        text = text_format % (bar.get_height())
        text_x = bar.get_x() + bar.get_width() / 2
        color = 'white'
        va = 'center'
        text_y = bar.get_y() + bar.get_height() / 2
        if bar.get_window_extent(renderer).height < 10:
            color = 'black'
            va = 'bottom'
            text_y = bar.get_y() + 1.5 * bar.get_height()

        ax.text(text_x, text_y, text, ha='center', va=va, color=color, **kwargs)

    plt.ylim([0, 1.10])

# https://matplotlib.org/3.1.1/gallery/units/bar_unit_demo.html
def comparison_bar_chart(data_set_name, scoring, labels, legend_labels, train, test, text_format='%.2f', **kwargs):
    fig, ax = plt.subplots()
    ind = np.arange(len(train))
    width = 0.4
    ax.barh(ind, train, width, label=legend_labels[0]) #, yerr=score_stds)
    ax.barh(ind + width, test, width, label=legend_labels[1]) #, yerr=tuned_score_stds)
    ax.set_xlabel(label(scoring))
    ax.set_title('%s for %s' % (label(scoring), data_set_name))
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(model_names)

    ax.legend(loc='upper left')
    ax.autoscale_view()
    ax.invert_yaxis()
    label_bars('%.2f')

    # Based on https://stackoverflow.com/a/50354131
    fig = plt.gcf()
    bars = ax.patches
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for bar in bars:
        text = text_format % (bar.get_width())
        color = 'white'
        va = 'center'
        ha = 'center'
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + bar.get_height() / 2
        if bar.get_window_extent(renderer).width < 10:
            color = 'black'
            ha = 'left'
            text_y = bar.get_y() + 1.5 * bar.get_width()

        ax.text(text_x, text_y, text, ha=ha, va=va, color=color, **kwargs)

# https://matplotlib.org/examples/lines_bars_and_markers/barh_demo.html
def simple_bar_chart(labels, data, text_format='%1.2f', **kwargs):
    ax = plt.gca()
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, data)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    label_bars(text_format)

if __name__ == "__main__":
    all_data_sets = [phishing, titanic, spambase, bank, dexter, breast_cancer, hypo_thyroid, mushrooms, polish_bankruptcy, online_news, occupancy, placement, pulsars, credit_card_default, pima]
    all_models = [decision_tree, ada_boost, mlp, svm_linear, svm_rbf, knn]
    selected_data_sets = [dexter, polish_bankruptcy] #, hypo_thyroid, dexter, polish_bankruptcy] #[polish_bankruptcy, pima, credit_card_default, titanic, spambase]
    selected_models = all_models # [mlp, perceptron, svm_linear, svm_rbf, knn] #all_models
    data_set_metrics = ['balanced_accuracy'] # ['f1', 'roc_auc', 'balanced_accuracy', 'average_precision']
    extra_metrics = ['sensitivity', 'specificity']
    for metric in data_set_metrics:
        scoring = metric
        scoring_name = label(scoring)
        print('Evaluating %s' % (scoring_name))
        for data_set in selected_data_sets:
            if data_set == polish_bankruptcy:
                num_folds = 20
            elif data_set == dexter:
                num_folds = 10
            print('Number of folds = %s' % (num_folds))
            data_set_name = label(data_set)
            print('Evaluating %s' % (data_set_name))
            X, y = data_set()
            print('Bin counts', np.bincount(y))
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

            # Limit size
            X_train, y_train = limit_size(data_set_name + ' training', X_train, y_train, size_limit)
            X_test, y_test = limit_size(data_set_name + ' testing', X_test, y_test, size_limit)

            # Rebalance
            #X_train, y_train = rebalance(data_set_name + ' training', X_train, y_train)
            #X_test, y_test = rebalance(data_set_name + ' testing', X_test, y_test, np.min)

            # SMOTE
            X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
            # Don't use SMOTE on test for average_precision. Only for training.
            #X_test, y_test = SMOTE().fit_resample(X_test, y_test)

            plt.figure(11, figsize=(7, 5))
            plt.title('Learning times for %s' % (data_set_name))
            plt.xlabel('Data Set Size')
            plt.ylabel('Time (s)')

            plt.figure(12, figsize=(7, 5))
            plt.title('Query times for %s' % (data_set_name))
            plt.xlabel('Data Set Size')
            plt.ylabel('Time (s)')

            """
            plt.figure(2, figsize=(7, 5))
            plt.title('Initial Precision Recall for %s' % (data_set_name))
            majority_class = len(y_test[y_test==1]) / len(y_test)
            plt.plot([0, 1], [majority_class, majority_class], linestyle='--', label='No skill')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.45, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')

            plt.figure(4, figsize=(7, 5))
            plt.title('Optimized Precision Recall for %s' % (data_set_name))
            majority_class = len(y_test[y_test==1]) / len(y_test)
            plt.plot([0, 1], [majority_class, majority_class], linestyle='--', label='No skill')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.45, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            """

            plt.figure(1, figsize=(7, 5))
            plt.title('Initial ROC AUC for %s' % (data_set_name))
            plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='Random guessing')
            plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='Perfect performance')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')

            plt.figure(3, figsize=(7, 5))
            plt.title('Optimized ROC AUC for %s' % (data_set_name))
            plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='Random guessing')
            plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='Perfect performance')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')

            tuned_train_scores = []
            tuned_test_scores = []
            test_scores = []
            train_scores = []
            improvements = []
            learning_times = []
            query_times = []
            extra_train_scores = []
            extra_test_scores = []
            extra_tuned_train_scores = []
            extra_tuned_test_scores = []
            for model in selected_models:
                print('Evaluating %s' % (model_label(model)))
                estimator, tuned_estimator, orig_train_score, orig_test_score, tuned_train_score, \
                    tuned_test_score, learning_time, query_time \
                    = evaluate_model(data_set_name, model, X_train, y_train, X_test, y_test)
                train_scores.append(orig_train_score)
                test_scores.append(orig_test_score)
                tuned_train_scores.append(tuned_train_score)
                tuned_test_scores.append(tuned_test_score)
                improvements.append(tuned_test_score - orig_test_score)
                learning_times.append(learning_time)
                query_times.append(query_time)
                print(model_label(model), 'learning_time', learning_time, 'query_time', query_time)
                print(model_label(model), 'orig', orig_test_score, 'improved', tuned_test_score)
                for e, train, test in [(estimator, extra_train_scores, extra_test_scores), (tuned_estimator, extra_tuned_train_scores, extra_tuned_test_scores)]:
                    curr_extra_train_scores = {}
                    curr_extra_test_scores = {}
                    for extra_metric in extra_metrics:
                        train_score, test_score = score_train_and_test(e, extra_metric, X_train, y_train, X_test, y_test, 0, model)
                        curr_extra_train_scores[extra_metric] = train_score
                        curr_extra_test_scores[extra_metric] = test_score
                    train.append(curr_extra_train_scores)
                    test.append(curr_extra_test_scores)

            plt.figure(1)
            plt.legend()
            save_fig('initial_roc_auc_curve', data_set_name, width=6, height=6)

            plt.figure(3)
            plt.legend()
            save_fig('optimized_roc_auc_curve', data_set_name, width=6, height=6)
            
            """
            plt.figure(2)
            plt.legend()
            save_fig('initial_precision_recall_curve', data_set_name)

            plt.figure(4)
            plt.legend()
            save_fig('optimized_precision_recall_curve', data_set_name)
            """

            model_names = [model_label(x) for x in selected_models]

            plt.figure(5)
            simple_bar_chart(model_names, learning_times)
            plt.title('Time to learn for %s' % (data_set_name))
            plt.ylabel('Model')
            plt.xlabel('Time (s)')
            save_fig('training_times_bar_chart', data_set_name)

            plt.figure(6)
            simple_bar_chart(model_names, query_times, '%1.3f')
            plt.title('Time to query for %s' % (data_set_name))
            plt.ylabel('Model')
            plt.xlabel('Time (s)')
            save_fig('query_times_bar_chart', data_set_name)

            plt.figure(7)
            simple_bar_chart(model_names, improvements)
            plt.title('Hyperparameter tuning improvements for %s' % (data_set_name))
            plt.ylabel('Model')
            plt.xlabel('Delta ' + scoring_name)
            save_fig('improvements', data_set_name)

            plt.figure(8)
            comparison_bar_chart(data_set_name, scoring, model_names, ['Test', 'Train'], test_scores, train_scores)
            save_fig('scores', data_set_name)

            plt.figure(9)
            comparison_bar_chart(data_set_name, scoring, model_names, ['Test', 'Train'], tuned_test_scores, tuned_train_scores)
            save_fig('tuned_scores', data_set_name)

            plt.figure(10)
            comparison_bar_chart(data_set_name, scoring, model_names, ['Original', 'Tuned'], test_scores, tuned_test_scores)
            save_fig('before_and_after', data_set_name)

            """
            plt.figure(11)
            plt.legend()
            save_fig('training_times', data_set_name)

            plt.figure(12)
            plt.legend()
            save_fig('query_times', data_set_name)
            """

            for title, train, test in [('Initial', extra_train_scores, extra_test_scores), ('Tuned', extra_tuned_train_scores, extra_tuned_test_scores)]:
                for extra_metric in extra_metrics:
                    fig_num += 1
                    metric_train_scores = [x[extra_metric] for x in train]
                    metric_test_scores = [x[extra_metric] for x in test]
                    plt.figure(fig_num)
                    comparison_bar_chart(data_set_name, extra_metric, model_names, ['Train', 'Test'], metric_train_scores, metric_test_scores)
                    plt.title(title + ' ' + extra_metric + ' for ' + data_set_name)
                    save_fig(title + '_' + extra_metric, data_set_name)
