#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 06:27:09 2020

@author: brandon
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import make_classification, make_hastie_10_2, make_moons, make_circles, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import scipy.sparse
from sklearn.tree import plot_tree

#from skhubness.data import load_dexter

import matplotlib.pyplot as plt

random_state = 0

base_dir = '/home/brandon/ml/data_sets/'

def rebalance(name, X, y, f=np.max):
    bins = np.bincount(y)
    print('bins for %s: %s' % (name, str(bins)))
    class_size = f(bins)
    X_balanced, y_balanced = np.empty((0, X.shape[1])), np.empty((0), dtype=int)
    for i in range(bins.size):
        X_upsampled, y_upsampled = resample(
                X[y == i],
                y[y == i],
                replace=True,
                n_samples=class_size,
                random_state=random_state)
        X_balanced = np.vstack((X_balanced, X_upsampled))
        y_balanced = np.hstack((y_balanced, y_upsampled))

    print('balanced bins for %s: %s' % (name, str(np.bincount(y_balanced))))
    return X_balanced, y_balanced

def limit_size(name, X, y, size):
    if y.size < size:
        return X, y
    X_limited, y_limited = np.empty((0, X.shape[1])), np.empty((0), dtype=int)
    bins = np.bincount(y)
    for i in range(bins.size):
        if bins[i] > size:
            X_downsampled, y_downsampled = resample(
                    X[y == i],
                    y[y == i],
                    replace=True,
                    n_samples=size,
                    random_state=random_state)
        else:
            X_downsampled = X[y == i]
            y_downsampled = y[y == i]
        X_limited = np.vstack((X_limited, X_downsampled))
        y_limited = np.hstack((y_limited, y_downsampled))
    print('limited bins for %s: %s' % (name, str(np.bincount(y_limited))))
    return X_limited, y_limited

def dexter():
    sparse_matrix = scipy.sparse.load_npz('data/dexter/dexter.npz')
    X = sparse_matrix[:, :-1].toarray().astype(np.int16)
    y = sparse_matrix[:, -1].toarray().astype(np.int8).reshape(-1)
    print(y)
    print('Dexter bin counts', np.bincount(y))
    return X, y

def dexter_csv():
    df = pd.read_csv('data/dexter/dexter.csv')
    print(df.head())
    X_df = df.iloc[:, :-1]
    X = X_df.values

    y_df = df.iloc[:, -1]
    y = y_df.values
    y = y.astype(np.int64)
    print(y)
    print('Dexter bin counts', np.bincount(y))
    return X, y

def dexter_old():
    X, y = load_dexter()
    print(X)
    print(y)
    y = y.astype(np.int64)
    y[y == -1] = 0
    print(y)
    print('Dexter bin counts', np.bincount(y))
    return X, y

# Great separation between perceptron, decision tree, and knn
# SVM has very poor precision/recall, ada_boost has high precision recall.
# Others are similar.
# KNN and ada_boost perform highly after optimization
def polish_bankruptcy():
    #df = pd.read_csv(base_dir + 'polish_bankruptcy/5year.arff', encoding='utf-8', header=None, na_values=['?'])
    df = pd.read_csv('data/polish_bankruptcy/5year.arff', encoding='utf-8', header=None, na_values=['?'])
    df.fillna(df.mean(), inplace=True)
    print(df.head())
    X_df = df.iloc[:, :-1]
    X = X_df.values

    y_df = df.iloc[:, -1]
    y = y_df.values
    #X, y = rebalance('bank', X, y, f=np.min)

    return X, y
"""
mlp = {'mlpclassifier__activation': 'logistic',
       'mlpclassifier__hidden_layer_sizes': (10,),
       'mlpclassifier__solver': 'adam'}
knn = {'kneighborsclassifier__metric': 'manhattan',
       'kneighborsclassifier__n_neighbors': 19,
       'kneighborsclassifier__weights': 'distance'}
decision_tree = {'decisiontreeclassifier__criterion': 'entropy',
                 'decisiontreeclassifier__min_impurity_decrease': 0.01}
gradient_boosting = {'gradientboostingclassifier__criterion': 'friedman_mse',
                     'gradientboostingclassifier__learning_rate': 0.2,
                     'gradientboostingclassifier__loss': 'deviance',
                     'gradientboostingclassifier__max_depth': 8,
                     'gradientboostingclassifier__max_features': 'log2',
                     'gradientboostingclassifier__min_samples_leaf': 0.1,
                     'gradientboostingclassifier__min_samples_split': 0.1,
                     'gradientboostingclassifier__n_estimators': 10,
                     'gradientboostingclassifier__subsample': 0.95}
svc = {'svc__C': 10.0,
       'svc__gamma': 0.001,
       'svc__kernel': 'rbf'}
{'svc__gamma': 0.04529533767703873, 'svc__C': 11.524828341803946}
{'svc__gamma': 0.05492116483887789, 'svc__C': 7566.218500481047}
"""


def wifi():
    df = pd.read_csv(base_dir + 'wifi/wifi_localization.txt', sep='\t', header=None, encoding='utf-8')
    X_df = df.iloc[:, :-1]
    X = X_df.values
    #print('X',  X_df.head())

    y_df = df.iloc[:, -1]
    #print(y_df)
    y_df = y_df.map({1: 0, 2: 1, 3: 0, 4: 1})
    y = y_df.values.astype(np.bool)
    print('Wifi bin counts', np.bincount(y))
    return X, y

def qsar_androgen():
    df = pd.read_csv(base_dir + 'qsar_androgen/qsar_androgen_receptor.csv', sep=';', header=None, encoding='utf-8')

    X_df = df.iloc[:, :-1]
    print('X',  X_df.head())
    X = X_df.values.astype(np.bool)

    y_df = df.iloc[:, 1024]
    mapping = {'negative': 0,
               'positive': 1}
    #print('before', y_df.head())
    y_df = y_df.map(mapping)

    y = y_df.values.astype(np.bool)
    print('Default bin counts', np.bincount(y))

    return X, y

def breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return X, y

# 'scores': {'perceptron': (0.9082, 0.9962,
#            'decision_tree': (0.7676, 0.9532),
#            'knn': (0.894, 0.983),
#            'svm': (0.9415, 0.9981),
#            'ada_boost': (0.9072, 0.984)}
def synthetic1():
    X, y = make_classification(n_samples=1500, n_features=20, n_informative=10, n_redundant=5, n_repeated=2, random_state=random_state)
    print('Bin counts for synthetic1', np.bincount(y))
    return X, y

def synthetic2():
    X, y = make_classification(n_samples=1500, n_features=5, n_informative=5, n_redundant=0, n_repeated=0, random_state=random_state)
    print('Bin counts for synthetic2', np.bincount(y))
    return X, y

# knn outperforms others
def half_moons():
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=random_state)
    return X, y

# Really breaks svm_linear and perceptron
def circles():
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=random_state)
    return X, y

# Decision tree and KNN struggle on this data set, but others are similar.
# Perceptron doesn't converge by default.
def pima():
    df = pd.read_csv(base_dir + 'pima_diabetes/diabetes.csv', encoding='utf-8')

    X_df = df.iloc[:, :-1]
    print('X',  X_df.head())
    X = X_df.values

    y_df = df.iloc[:, -1]
    y = y_df.values
    print('Default bin counts', np.bincount(y))

    return X, y

# Hard problem, but algorithms perform similarly
# ROC: ada_boost, perceptron, SVM, KNN, decision_tree
# PR: ada_boost, (perceptron, SVM), (KNN, decision_tree)
def credit_card_default():
    df = pd.read_csv(base_dir + 'credit_card_default/UCI_Credit_Card.csv', encoding='utf-8')

    X_df = df.iloc[:, :-1]
    print('X',  X_df.head())
    X = X_df.values

    y_df = df.iloc[:, -1]
    y = y_df.values
    print('Default bin counts', np.bincount(y))

    X, y = rebalance('default', X, y, f=np.min)

    return X, y


# PR: (ada_boost, svm), (perceptron, decision_tree, knn)
# ROC: (ada_boost, perceptron), (svm, knn), decision_tree
def pulsars():
    df = pd.read_csv(base_dir + 'pulsars/pulsar_stars.csv', encoding='utf-8')

    X_df = df.iloc[:, :-1]
    print('X',  X_df.head())
    #X_df = pd.get_dummies(X_df, drop_first=True)
    X = X_df.values

    y_df = df['target_class']
    y = y_df.values
    print('Pulsar bin counts', np.bincount(y))

    X, y = rebalance('pulsars', X, y, f=np.min)

    return X, y


# Good separation, but so little data
def placement():
    df = pd.read_csv(base_dir + 'placement/Placement_Data_Full_Class.csv', encoding='utf-8')

    X_df = df.iloc[:, :-2]
    print('X',  X_df.head())
    X_df = pd.get_dummies(X_df, drop_first=True)
    X = X_df.values

    y_df = df['status']
    mapping = {'Not Placed': 0,
               'Placed': 1}
    #print('before', y_df.head())
    y_df = y_df.map(mapping)
    #print('after', y_df.head())
    y = y_df.values
    print('Placement bin counts', np.bincount(y))
    return X, y

# Way too easy
def occupancy():
    df1 = pd.read_csv(base_dir + 'occupancy/datatraining.txt', encoding='utf-8')
    df2 = pd.read_csv(base_dir + 'occupancy/datatest.txt', encoding='utf-8')
    df3 = pd.read_csv(base_dir + 'occupancy/datatest2.txt', encoding='utf-8')
    df = pd.concat([df1, df2, df3])
    df = df.drop(columns=['date', 'Temperature'])

    X_df = df.iloc[:, :-1]
    print('X', X_df.head())
    X = X_df.values

    y_df = df['Occupancy']
    y = y_df.values
    print('Occupancy bin counts', np.bincount(y))
    return X, y


# Very difficult for all learners
def online_news():
    df = pd.read_csv(base_dir + 'OnlineNewsPopularity/OnlineNewsPopularity_NoSpaces.csv', encoding='utf-8')
    df = df.drop(columns=['url', 'timedelta'])

    X_df = df.iloc[:, :-1]
    print('X', df.head())
    X = X_df.values

    y_df = df['shares']
    #print('bin counts', np.bincount(np.log10(y_df.values)))

    """
    plt.figure()
    plt.hist(y_df.values, bins = np.logspace(np.log10(y_df.min()), np.log10(y_df.max()), 1000))
    plt.gca().set_xscale("log")
    plt.show()
    """
    #print('histogram of online_news', np.histogram(y_df.values), bins=np.logspace(np.log10(1),np.log10(100000), 10))
    threshold = y_df.median()
    print('threshold is number of shares > %d' % (threshold))
    y = np.zeros(y_df.shape, dtype=int)
    y[y_df > threshold] = 1

    #X, y = resample(X, y, n_samples=3000, replace=True, random_state=random_state)
    return X, y

# Impossible to predict gender well. Not sure why AUC is so much better than accuracy
def abalone():
    df = pd.read_csv(base_dir + 'abalone/abalone.data',
                     header=None, encoding='utf-8')
    df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                  'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

    X_df = df.iloc[:, 1:]
    print(X_df.head())
    X = X_df.values

    y_df = df['Sex']
    # TODO: remove 'I'
    le = LabelEncoder()
    le.fit(y_df.values)
    y = le.transform(y_df.values)
    # Predicting gender doesn't work
    print('Sex classes', le.classes_)
    return X, y
    # One hot encoding of Sex
    #Y_df = pd.get_dummies(Y_df, drop_first=True)

# Too easy for all estimators, without removing many attributes
def mushrooms():
    df = pd.read_csv(base_dir + 'poisonous-mushrooms/expanded.csv', #'poisonous-mushrooms/agaricus-lepiota.data',
                     header=None, encoding='utf-8')
    df.columns = ['edible',
                  'cap-shape',
                  'cap-surface',
                  'cap-color',
                  'bruises',
                  'odor',
                  'gill-attachment',
                  'gill-spacing',
                  'gill-size',
                  'gill-color',
                  'stalk-shape',
                  'stalk-root',
                  'stalk-surface-above-ring',
                  'stalk-surface-below-ring',
                  'stalk-color-above-ring',
                  'stalk-color-below-ring',
                  'veil-type',
                  'veil-color',
                  'ring-number',
                  'ring-type',
                  'spore-print-color',
                  'population',
                  'habitat']
    print(df.head())

    X_df = df.iloc[:, 1:]
    X_df.drop(['stalk-root'], axis=1)
    print(X_df.head())
    X_df = pd.get_dummies(X_df, drop_first=True)
    print(X_df.head())
    X = X_df.values

    y_df = df['edible']
    le = LabelEncoder()
    le.fit(y_df.values)
    y = le.transform(y_df.values)
    print(y_df.head())
    return X, y

# Too easy for too many learners
def hypo_thyroid():
    df = pd.read_csv(base_dir + 'thyroid-disease/allhypo.clean.csv',
                 header=None, encoding='utf-8', na_values=['?'])
    #print(df.head())
    df.drop(df.columns[27],axis=1,inplace=True)
    df.fillna(df.mean(), inplace=True)
    #print(df.head())

    X_df = df.iloc[:, :-2]
    X_df = pd.get_dummies(X_df, drop_first=True)
    X = X_df.values
    #print(X_df.head())

    y_df = df.iloc[:, -1]
    mapping = {'negative.': 0,
               'compensated hypothyroid.': 1,
               'primary hypothyroid.': 1}
    print('before', y_df.head())
    y_df = y_df.map(mapping)
    """
    print('negative', y_df[y_df == 0].size)
    print('compensated', y_df[y_df == 1].size)
    print('primary', y_df[y_df == 2].size)
    print('secondary', y_df[y_df == 3].size)
    print('total', y_df.size)
    """
    y = y_df.values
    #print('thyroid', np.bincount(y))
    return X, y

# performance very similar for all models
# SVM has poor precision/recall, as does decision tree.
# Decision tree has poor ROC AUC
def bank():
    df = pd.read_csv(base_dir + 'bank/bank-additional/bank-additional-full.csv',
                 sep=';', quotechar='"', encoding='utf-8')
    print(df.head())
    df.fillna(df.mean(), inplace=True)
    #print(df.head())

    X_df = df.iloc[:, :-1]
    X_df = pd.get_dummies(X_df, drop_first=True)
    X = X_df.values

    y_df = df.iloc[:, -1]
    mapping = {'no': 0, 'yes': 1}
    y_df = y_df.map(mapping)
    y = y_df.values

    X, y = rebalance('bank', X, y, f=np.min)

    return X, y

# Good at showing differences between estimators
# ROC AUC: ada_boost, (perceptron, svm, knn), decision_tree
# PR avg: ada_boost, (perceptron, svm, decision_tree), knn
def spambase():
    df = pd.read_csv(base_dir + 'spambase/spambase.data',
             header=None, encoding='utf-8')
    X_df = df.iloc[:, :-1]
    X = X_df.values

    y_df = df.iloc[:, -1]
    y = y_df.values
    return X, y

# Good at showing differences between estimators
# PR: SVM, perceptron, ada_boost, decision_tree, knn
# PR(no smote): KNN, (SVM, ada_boost), perceptron, decision_tree
# ROC: (knn, ada_boost, SVM, perceptron), decision_tree
# ROC(no smote): (KNN, ada_boost), (svm, perceptron), decision_tree
# https://www.kaggle.com/schmitzi/titanic-data-cleaning
def titanic():
    df = pd.read_csv(base_dir + 'titanic/titanic-clean.csv', encoding='utf-8')
    X_df = df.iloc[:, 2:]
    X_df = pd.get_dummies(X_df, drop_first=True)
    #X_df['Fare'] = np.log(X_df['Fare'])
    #print(X_df.head())
    X = X_df.values

    y_df = df['Survived']
    y = y_df.values
    return X, y

# Too easy for models
def phishing():
    """
    @attribute having_IP_Address  { -1,1 }
    @attribute URL_Length   { 1,0,-1 }
    @attribute Shortining_Service { 1,-1 }
    @attribute having_At_Symbol   { 1,-1 }
    @attribute double_slash_redirecting { -1,1 }
    @attribute Prefix_Suffix  { -1,1 }
    @attribute having_Sub_Domain  { -1,0,1 }
    @attribute SSLfinal_State  { -1,1,0 }
    @attribute Domain_registeration_length { -1,1 }
    @attribute Favicon { 1,-1 }
    @attribute port { 1,-1 }
    @attribute HTTPS_token { -1,1 }
    @attribute Request_URL  { 1,-1 }
    @attribute URL_of_Anchor { -1,0,1 }
    @attribute Links_in_tags { 1,-1,0 }
    @attribute SFH  { -1,1,0 }
    @attribute Submitting_to_email { -1,1 }
    @attribute Abnormal_URL { -1,1 }
    @attribute Redirect  { 0,1 }
    @attribute on_mouseover  { 1,-1 }
    @attribute RightClick  { 1,-1 }
    @attribute popUpWidnow  { 1,-1 }
    @attribute Iframe { 1,-1 }
    @attribute age_of_domain  { -1,1 }
    @attribute DNSRecord   { -1,1 }
    @attribute web_traffic  { -1,0,1 }
    @attribute Page_Rank { -1,1 }
    @attribute Google_Index { 1,-1 }
    @attribute Links_pointing_to_page { 1,0,-1 }
    @attribute Statistical_report { -1,1 }
    @attribute Result  { -1,1 }
    """
    df = pd.read_csv(base_dir + 'phishing/phishing.data',
                     header=None, encoding='utf-8')
    df.columns = ['having_IP_Address',
                'URL_Length ',
                'Shortening_Service',
                'having_At_Symbol',
                'double_slash_redirecting',
                'Prefix_Suffix',
                'having_Sub_Domain',
                'SSLfinal_State',
                'Domain_registeration_length',
                'Favicon',
                'port',
                'HTTPS_token',
                'Request_URL',
                'URL_of_Anchor',
                'Links_in_tags',
                'SFH',
                'Submitting_to_email',
                'Abnormal_URL',
                'Redirect',
                'on_mouseover',
                'RightClick',
                'popUpWindow',
                'Iframe',
                'age_of_domain',
                'DNSRecord ',
                'web_traffic',
                'Page_Rank',
                'Google_Index',
                'Links_pointing_to_page',
                'Statistical_report',
                'Result']

    X_df = df.iloc[:, :-1]
    print(X_df.head())
    X = X_df.values

    y_df = df['Result']
    mapping = {-1: 0,
               1: 1}
    #print('before', y_df.head())
    y_df = y_df.map(mapping)
    y = y_df.values
    return X, y

# TODO: Kernel density estimate of all parameters
# TODO: Show correlation with y
# TODO: Correlation heatmap

if __name__ == "__main__":
    from project1 import save_fig, label, get_path
    np.set_printoptions(suppress=True)

    # https://stackoverflow.com/a/29432741
    def plot_corr(data_set_name, df):
        f = plt.figure()
        plt.matshow(df.corr(), fignum=f.number)
        #plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
        #plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix for %s' %(data_set_name))
        save_fig('correlation_heatmap', data_set_name)

    def common_analysis(data_set_name, X, y, doFeatureCorrelations=False):
        n_features = X.shape[1]
        print('Number of features', n_features)
        print('Bin counts', np.bincount(y))

        dt = DecisionTreeClassifier(class_weight='balanced', max_features=None)
        dt.fit(X, y)
        feature_importances = np.sort(dt.feature_importances_)[::-1]
        feature_importances = feature_importances[feature_importances > 0]
        print('Feature importances', np.sum(feature_importances > 0), np.round(feature_importances, 4))
        #print('Features used', len(np.unique(dt.tree_.feature)))
        
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)
        corrs = []
        col_corrs = []
        for name, col in X_df.iteritems():
            corr = col.corr(y_df.iloc[:, 0])
            corrs.append(corr)
            #print(name, col.shape, y_df.shape, '%.3f' % (corr))
            #col.hist()
            #col.plot.kde()
            #plt.show()
            if doFeatureCorrelations:
                for name2, col2 in X_df.iteritems():
                    if name < name2:
                        corr2 = col2.corr(col)
                        col_corrs.append(corr2)
        if doFeatureCorrelations:
            total = n_features * (n_features - 1) / 2
            for i in np.linspace(start=0.1, stop=0.9, num=9):
                count = np.sum(np.abs(col_corrs) > i)
                print('feature corr with each other > ' + str(i), count, '%.3f' % (count / total))

            #plot_corr(data_set_name, X_df)

        for i in np.linspace(start=0.1, stop=0.9, num=9):
            print('label correlation > ' + str(i), np.sum(np.abs(corrs) > i))
        corrs.sort(reverse=True)
        #print(corrs)
        pd.DataFrame(corrs).hist(log=(len(corrs) > 100))
        ax = plt.gca()
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Count of features')
        plt.title('Histogram of %s feature correlations with Y' % (data_set_name))
        save_fig('feature_correlations_with_y', data_set_name)

        if doFeatureCorrelations:
            pd.DataFrame(col_corrs).hist(log=(len(col_corrs) > 100))
            plt.title('Histogram of %s feature correlations with each other' % (data_set_name))
            ax = plt.gca()
            ax.set_xlabel('Correlation')
            ax.set_ylabel('Count of features')
            save_fig('feature_correlations', data_set_name)

    def plot_dt_tree(dt, data_set_name):
        plt.gcf().set_size_inches(50, 30)
        plot_tree(dt, fontsize=10)
        plt.savefig(get_path('decision_tree', data_set_name, None) + '.png')
        plt.close()
        print('Features used', len(np.unique(dt.tree_.feature)))

    def analyze_polish_bankruptcy():
        X, y = polish_bankruptcy()
        data_set_name = label(polish_bankruptcy)

        dt = DecisionTreeClassifier(class_weight='balanced', max_depth=12, max_leaf_nodes=175)
        dt.fit(X, y)
        plot_dt_tree(dt, data_set_name)

        common_analysis(data_set_name, X, y, True)
        


        """
        X = StandardScaler().fit_transform(X)
        lr = LogisticRegression(class_weight='balanced')
        lr.fit(X, y)
        coef = np.sort(np.abs(lr.coef_))
        print('Linear coefficients', np.round(coef[0][::-1], 4))
        """

        X_df = pd.DataFrame(X)
        ax = X_df.skew().hist()
        ax.set_xlabel('Skew')
        ax.set_ylabel('Count of features')
        plt.title('Histogram of skew for ' + data_set_name)
        save_fig('skew', data_set_name)
        ax = X_df.kurt().hist()
        ax.set_xlabel('Kurtosis')
        ax.set_ylabel('Count of features')
        plt.title('Histogram of kurtosis for ' + data_set_name)
        save_fig('kurtosis', data_set_name)
        for name, col in X_df.iteritems():
            ax = col.hist(log=True)
            ax.set_xlabel('X%s' % (name))
            ax.set_ylabel('Number of samples')
            plt.title('Histogram for %s[X%s]' % (data_set_name, name))
            save_fig('histograms_%s' % (name), data_set_name)
        print('Finished for ' + data_set_name)

    def analyze_dexter():
        data_set_name = label(dexter)
        X, y = dexter()
        
        dt = DecisionTreeClassifier(class_weight='balanced', max_depth=4, max_leaf_nodes=5)
        dt.fit(X, y)
        plot_dt_tree(dt, data_set_name)

        X_df = pd.DataFrame(X)
        # Remove non-zero columns
        # https://stackoverflow.com/a/21165116
        X_df[X_df > 0] = 1
        X_df = X_df.loc[:, (X_df != 0).any(axis=0)]
        print('After removing zero columns, shape is', X_df.shape)
        y_df = pd.DataFrame(y)

        common_analysis(data_set_name, X_df.values, y_df.values[:, 0], True)

        match = np.zeros(X.shape[1])
        counts = np.zeros(X.shape[1])
        for col in range(X.shape[1]):
             match[col] = np.sum((X[:, col] == y))
             counts[col] = np.sum(X[:, col])
        match = np.sort(match)[::-1]
        #print('match counts', match[0:100], match[-100:])
        print('match counts ', np.sum(match > 160), np.sum(match < 140))
        counts = np.sort(counts)[::-1]
        print('counts', counts[0:100])

    analyze_polish_bankruptcy()
    analyze_dexter()

    """
    selected_data_sets = [polish_bankruptcy, dexter] # [polish_bankruptcy, pima, bank, circles, credit_card_default, titanic, spambase]
    for data_set in selected_data_sets:
        data_set_name = label(data_set)
        print('Exploring %s' % (data_set_name))





        corrs = []
        col_corrs = []
        for name, col in X_df.iteritems():
            corr = col.corr(y_df.iloc[:, 0])
            corrs.append(corr)
            print(name, col.shape, y_df.shape, '%.3f' % (corr))
            #col.hist()
            #col.plot.kde()
            #plt.show()
            for name2, col2 in X_df.iteritems():
                if name < name2:
                    corr2 = col2.corr(col)
                    col_corrs.append(corr2)
        plot_corr(data_set_name, X_df)

        corrs.sort(reverse=True)
        print(corrs)
        pd.DataFrame(corrs).hist()
        plt.title('Histogram of %s feature correlations with Y' % (data_set_name))
        save_fig('feature_correlations_with_y', data_set_name)


        col_corr_df = pd.DataFrame(col_corrs)
        col_corr_df.hist()
        plt.title('Histogram of %s feature correlations with each other' % (data_set_name))
        save_fig('feature_correlations', data_set_name)
        #print(col_corr_df.plot.kde())
        # https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
        # sns.pairplot(df)
        # pd.plotting.scatter_matrix(df, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
        #corr = df.corr()
        #print(corr)
        #corr.style.background_gradient(cmap='coolwarm').set_precision(2)
        #fig, ax = plt.subplots(figsize=df.shape)
        #ax.matshow(corr)
        #plt.xticks(range(len(corr.columns)), corr.columns)
        #plt.yticks(range(len(corr.columns)), corr.columns)
        #plt.show()
        """