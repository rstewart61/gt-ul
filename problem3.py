#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 07:47:14 2020

@author: brandon
"""

import math
from enum import Enum
from copy import deepcopy
import pandas as pd
from sklearn.model_selection import train_test_split, validation_curve, RandomizedSearchCV
import numpy as np
from sklearn.utils import resample
import scipy.sparse
from sklearn.metrics import balanced_accuracy_score, silhouette_score, adjusted_mutual_info_score, completeness_score, homogeneity_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection, johnson_lindenstrauss_min_dim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.base import clone

from sklearn.manifold import TSNE
import seaborn as sns

from scipy.stats import kurtosis

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import joblib
import time
import os

RANDOM_STATE = 0
SIZE_LIMIT = 2000
LINE_WIDTH = 3
FIG_WIDTH = 6
FIG_HEIGHT = 3
fig_num = 0

K_MEANS_RESTARTS=32
K_MEANS_ITERATIONS=25
K_MEANS_MAX_ITER=300
EM_ITERATIONS=10
RP_ITERATIONS=10
ICA_ITERATIONS=10
EM_RESTARTS=20

TUNING_GRANULARITY=1000

# TODO: Cumulative explained variance for PCA (Python Machine Learning).
# TODO: Use LDA (or KPCA) rather than DT.

#TODO: research what AIC and BIC really mean. Both point to 2 clusters,
#       and the result can be evaluated with adjusted_rand_score

# NN: if clustering is really trivial, it's possible to add the cluster labels
#     to the original features to see if that improves the NN. Otherwise, just
#     use the cluster labels for the NN. There is no assumption that the
#     NN performance will improve. Just need to properly analyze results.


# Visualization:
# TSNE - visualize higher number of dimensions in 2 or 3.
# Can stick to first three to see if they provide intuition.
# Not looking for specific plots, but need at three plots to inform analysis.
# There are metrics that show how dense and how far apart clusters are. Can use
# these instead of just pair plotting first two features.
# Don't need visualizations for all 16 combinations, but at least some should be
# visualized, ones that you can make something out of (interesting).

# Should retune NN, but don't need model complexity analysis. Should talk about
# how tuning was done, however.

# See if DR plus clustering captures the original data while reducing noise and
# reducing convergence time.

# For clustering, we should see if cluster labels make sense, that is, tell us
# something about the data that we expected. For example, does the Euclidean
# distance make sense in helping us understand the data?

# CANNOT USE LABELS DURING LEARNING, but can use during evaluation. AMI, completeness, etc.
# NUMBER OF CLUSTERS CANNOT BE BASED ON LABELS. Don't use AMI and similar.
# Instead use things like SSE, Silhouette metric, BIC to choose number of clusters.
# NEED TWO ANALYSIS:
# 1) Should have one analysis to choose K.
# 2) Then another to evaluate the performance of the clustering.
# Must explain the clustering, especially if cluster labels don't align with ground truth labels.

# DT and LDA can be used for dim red on fourth DR.

# Should have a hold-out, if possible, that's not use for DR and clustering, then
# use this test set during NN evaluation. If not possible, at least mention the bias.

#pd.set_option('display.max_columns', None)

def label(d):
    if callable(d):
        d = d.__name__
    return d.replace('_', ' ').title()

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
                random_state=RANDOM_STATE)
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
                    random_state=RANDOM_STATE)
        else:
            X_downsampled = X[y == i]
            y_downsampled = y[y == i]
        X_limited = np.vstack((X_limited, X_downsampled))
        y_limited = np.hstack((y_limited, y_downsampled))
    print('limited bins for %s: %s' % (name, str(np.bincount(y_limited))))
    return X_limited, y_limited

class Analysis(Enum):
    DEFAULT = 'Default'
    KM = 'K Means'
    EM = 'Expectation Maximization'
    PCA = 'Principal Component Analysis'
    RP = 'Randomized Projection'
    ICA = 'Independent Component Analysis'
    DT = 'Decision Tree'
    LDA = 'Linear Discriminant Analysis'

class Config:
    def __init__(self, min_k=None, max_k=None,
                 best_k=None, best_features=None,
                 max_nn_nodes=None, best_nn_nodes=None):
        self.min_k = min_k
        self.max_k = max_k
        self.best_k = best_k
        self.best_features = best_features
        self.max_nn_nodes = max_nn_nodes
        self.best_nn_nodes = best_nn_nodes
    
    def clone(self):
        return deepcopy(self)
    
    def merge(self, other):
        for field in vars(self):
            if vars(other)[field] is not None:
                vars(self)[field] = vars(other)[field]
    
    def get_num_k(self):
        return min(100, self.max_k - self.min_k)
    
    def __repr__(self):
        return "(min_k=%s, max_k=%s, num_k=%s, best_k=%s)" % \
            (self.min_k, self.max_k, self.get_num_k(), self.best_k)

def dexter():
    sparse_matrix = scipy.sparse.load_npz('data/dexter/dexter.npz')
    X = sparse_matrix[:, :-1].toarray().astype(np.int16)
    y = sparse_matrix[:, -1].toarray().astype(np.int8).reshape(-1)
    print(y)
    print('Dexter bin counts', np.bincount(y))

    # max_iter = 5
    # learning_rate_init = 50
    estimator = MLPClassifier(early_stopping=False, solver='sgd',
                              max_iter=500, hidden_layer_sizes=(10,),
                              activation='logistic', learning_rate_init=0.1,
                              random_state=RANDOM_STATE)
    param_grid = {
        'hidden_layer_sizes': list(zip(
                np.linspace(start=1, stop=100).astype(int)
                )),
        #'alpha': np.logspace(start=-1, stop=-8, num=tuning_granularity),
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #'tol': np.logspace(start=-2, stop=-10, num=tuning_granularity),
        'learning_rate_init': np.logspace(start=5, stop=-7, num=TUNING_GRANULARITY),
        'max_iter': np.linspace(start=1, stop=500, num=TUNING_GRANULARITY).astype(int)
    }

    # , best_features=[12915, 12609]
    config = {
            Analysis.DEFAULT: {
                Analysis.DEFAULT: Config(min_k=1, max_k=50, max_nn_nodes=100),
                Analysis.KM: Config(min_k=2, best_k=10),
                Analysis.EM: Config(min_k=2, max_k=30, best_k=23),
                Analysis.PCA: Config(best_k=20, max_k=240),
                Analysis.RP: Config(best_k=29, max_k=240),
                Analysis.ICA: Config(max_k=100, best_k=16), # best=3
                Analysis.DT: Config(),
                Analysis.LDA: Config(best_k=1, max_k=1),
            },
            Analysis.PCA: {
                Analysis.DEFAULT: Config(min_k=2, max_k=100, max_nn_nodes=50),
                Analysis.KM: Config(best_k=16),
                Analysis.EM: Config(best_k=16)
            },
            Analysis.RP: {
                Analysis.DEFAULT: Config(min_k=2, max_k=100, max_nn_nodes=50),
                Analysis.KM: Config(best_k=30),
                Analysis.EM: Config(best_k=55)
            },
            Analysis.ICA: {
                Analysis.DEFAULT: Config(min_k=2, max_k=100, max_nn_nodes=50),
                Analysis.KM: Config(best_k=16),
                Analysis.EM: Config(best_k=12)
            },
            Analysis.DT: {
                Analysis.DEFAULT: Config(min_k=2, max_k=50, max_nn_nodes=50),
                Analysis.KM: Config(best_k=20),
                Analysis.EM: Config(best_k=17)
            },
            Analysis.LDA: {
                Analysis.DEFAULT: Config(min_k=1, max_k=1, max_nn_nodes=50),
                Analysis.KM: Config(best_k=1),
                Analysis.EM: Config(best_k=1)
            },
       }
    return X, y, config, estimator, param_grid

def dexter_like_noise():
    X, y = make_blobs(300, 20000, centers=1, random_state=RANDOM_STATE)
    config = {
        Analysis.DEFAULT: {
            Analysis.DEFAULT: Config(min_k=2, max_k=50),
            Analysis.KM: Config()
        }
    }
    return X, y, config, None, None

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

    estimator = MLPClassifier(early_stopping=False, solver='sgd',
                              max_iter=200, hidden_layer_sizes=(35,),
                              activation='logistic', learning_rate_init=0.7,
                              random_state=RANDOM_STATE)
    param_grid = {
        'hidden_layer_sizes': list(zip(
                np.linspace(start=1, stop=100).astype(int)
                )),
        #'alpha': np.logspace(start=-1, stop=-8, num=tuning_granularity),
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #'tol': np.logspace(start=-2, stop=-10, num=tuning_granularity),
        'learning_rate_init': np.logspace(start=5, stop=-7, num=TUNING_GRANULARITY),
        'max_iter': np.linspace(start=1, stop=500, num=TUNING_GRANULARITY).astype(int)
    }

    # , best_features=[34, 26]
    config = {
            Analysis.DEFAULT: {
                Analysis.DEFAULT: Config(min_k=1, max_k=30, max_nn_nodes=50),
                Analysis.KM: Config(min_k=2, best_k=23),
                #Analysis.EM: Config(min_k=2, max_k=1000, best_k=284),
                Analysis.EM: Config(min_k=2, max_k=10, best_k=4),
                Analysis.PCA: Config(max_k=40, best_k=30),
                Analysis.RP: Config(max_k=40, best_k=25),
                Analysis.ICA: Config(max_k=60, best_k=36),
                Analysis.DT: Config(),
                Analysis.LDA: Config(max_k=1, best_k=1),
            },
            Analysis.PCA: {
                Analysis.DEFAULT: Config(min_k=2, max_k=50, max_nn_nodes=50),
                Analysis.KM: Config(best_k=43),
                Analysis.EM: Config(best_k=42)
            },
            Analysis.RP: {
                Analysis.DEFAULT: Config(min_k=2, max_k=50, max_nn_nodes=50),
                Analysis.KM: Config(best_k=35),
                Analysis.EM: Config(best_k=43)
            },
            Analysis.ICA: {
                Analysis.DEFAULT: Config(min_k=2, max_k=50, max_nn_nodes=50),
                Analysis.KM: Config(best_k=42),
                Analysis.EM: Config(best_k=40)
            },
            Analysis.DT: {
                Analysis.DEFAULT: Config(min_k=2, max_k=50, max_nn_nodes=50),
                Analysis.KM: Config(best_k=9),
                Analysis.EM: Config(best_k=4)
            },
            Analysis.LDA: {
                Analysis.DEFAULT: Config(min_k=1, max_k=1, max_nn_nodes=50),
                Analysis.KM: Config(best_k=1),
                Analysis.EM: Config(best_k=1)
            },
    }
    return X, y, config, estimator, param_grid

def blobs():
  return make_blobs(n_samples=400, centers=4,
                    cluster_std=0.60, random_state=RANDOM_STATE)  

def score(estimator, X, y):
    y_predict = estimator.predict(X)
    return balanced_accuracy_score(y, y_predict)

class Problem:
    def __init__(self, name, X, y, config, estimator, param_grid):
        self.name = name
        self.config = config
        self.estimator = estimator
        self.param_grid = param_grid
        
        do_scaling = True
        if X.dtype.kind in np.typecodes["AllInteger"] and \
            X.shape[1] == 1 and np.count_nonzero(np.bincount(X[:,0])) > 1:
            enc = OneHotEncoder(drop='first', sparse=False)
            X = enc.fit_transform(X)
            do_scaling = False
        
        bin_counts = np.bincount(y)
        print('Bin counts', bin_counts)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        # SMOTE
        if len(bin_counts) > 1 and bin_counts[0] != bin_counts[1]:
            X_train, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
        
        # Limit size
        X_train, y_train = limit_size(name + ' training', X_train, y_train, SIZE_LIMIT)
        X_test, y_test = limit_size(name + ' testing', X_test, y_test, SIZE_LIMIT)
        
        if do_scaling:
            scaler = StandardScaler()
            #scaler = RobustScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.data_hash = joblib.hash([self.X_train, self.y_train])
    
    def new(fn):
        X, y, config, estimator, param_grid = fn()
        return Problem(label(fn), X, y, config, estimator, param_grid)
    
    def clone(self, new_X):
        result = Problem(self.name, new_X, self.y, self.config, self.estimator, self.param_grid)
        return result

    def fit(self, params):
        # pass estimator parameters just to clear cache when needed
        param_hash = joblib.hash(self.estimator.get_params())
        self.estimator, self.fit_time = cache(fit, self, **{"param_hash": param_hash})
    
    def score_train(self):
        y_predict = self.estimator.predict(self.X_train)
        return balanced_accuracy_score(self.y_train, y_predict)

    def score_test(self):
        y_predict = self.estimator.predict(self.X_test)
        return balanced_accuracy_score(self.y_test, y_predict)
    
    def get_config(self, dr_analysis, analysis):
        result = self.config[dr_analysis][Analysis.DEFAULT].clone()
        if not dr_analysis in self.config or not analysis in self.config[dr_analysis]:
            print('Warning: no config for %s: %s - %s' % (self.name, dr_analysis.name, analysis.name))
            return None
        result.merge(self.config[dr_analysis][analysis])
        return result
    
    def get_range(self, start, stop):
        num = min(100, stop-start)
        result = np.unique(np.linspace(start=start, stop=stop, num=num).astype(int))
        return result
    
    def k_range(self, dr_analysis, analysis):
        config = self.get_config(dr_analysis, analysis)
        return self.get_range(config.min_k, config.max_k)
    
    def nn_range(self, dr_analysis, analysis):
        config = self.get_config(dr_analysis, analysis)
        return self.get_range(1, config.max_nn_nodes)

cache_data = {}
def cache(fn, problem, key=None, **kwargs):
    global cache_data
    cache_key_args = {}
    for key, value in kwargs.items():
        if isinstance(value, (list, np.ndarray)) and len(value) > 0:
            value = [value[0], value[-1], len(value)]
        cache_key_args[key] = value
    file_name = "cache/%s.%s.%s.%s.%s.dump" % (fn.__name__, problem.name, problem.data_hash, key, cache_key_args)
    
    if file_name in cache_data:
        print("loading memory cache for fn=%s, problem=%s, hash=%s, args=%s" % (fn.__name__, problem.name, problem.data_hash, cache_key_args))
        return cache_data[file_name]
    
    if os.path.exists(file_name):
        print("loading disk cache for fn=%s, problem=%s, hash=%s, args=%s" % (fn.__name__, problem.name, problem.data_hash, cache_key_args))
        cache_data[file_name] = joblib.load(file_name)
        return cache_data[file_name]
    
    print("Computing cache for fn=%s, problem=%s, hash=%s, args=%s" % (fn.__name__, problem.name, problem.data_hash, cache_key_args))
    start = time.time()
    data = fn(problem, **kwargs)
    result = (data, time.time() - start)
    print("Completed in %2.2f seconds" % (result[1]))
    joblib.dump(result, file_name, compress=3)
    return result

def format_number(x):
    if isinstance(x, int):
        return "%d" % x
    elif x > 0.1 and x < 10000:
        return "%.2f" % (x)
        #plt.gca().set_ylabel("%s (best=%.2f)" % (ylabel, best_y))
    elif x > 0.0001:
        return "%.5f" % (x)
        #plt.gca().set_ylabel("%s (best=%.5f)" % (ylabel, best_y))
    else:
        return "%1.3e" % (x)
        #plt.gca().set_ylabel("%s (best=%1.3e)" % (ylabel, best_y))
    

def label_best(problem, y, dr_analysis, analysis, color="tab:blue"):
    config = problem.get_config(dr_analysis, analysis)
    if not config.best_k:
        return
    x = problem.k_range(dr_analysis, analysis)
    
    xlabel = plt.gca().get_xlabel()
    plt.gca().set_xlabel("%s (best=%s)" % (xlabel, config.best_k))
    ylabel = plt.gca().get_ylabel()
    matching_indices = np.where(x == config.best_k)
    if len(matching_indices) == 0 or len(matching_indices[0]) == 0:
        print("Could not find %s in %s" % (config.best_k, x))
        return
    index = np.where(x == config.best_k)[0][0]
    best_y = y[index]
    plt.gca().set_ylabel("%s (best=%s)" % (ylabel, format_number(best_y)))

    plt.axvline(x=config.best_k, linestyle='--', color=color)

def plot_dir(problem, dr_analysis, analysis):
    return 'plots/%s/%s/%s' % (problem.name, dr_analysis.name, analysis.name)

def finalize_plot(problem, analysis, title, dr_analysis=Analysis.DEFAULT,
                  width=FIG_WIDTH, height=FIG_HEIGHT):
    plt.grid()
    plt.xticks(fontsize=12)
    plt.gcf().set_size_inches(width, height)
    path = '%s/%s.png' % (plot_dir(problem, dr_analysis, analysis), title)
    print('Saving plot to %s' % (path))
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def fit(problem, **kwargs):
    problem.estimator.fit(problem.X_train, problem.y_train)
    return problem.estimator

def train_kmeans(problem, **kwargs):
    km = KMeans(**kwargs)
    km.fit(problem.X_train)
    return km

def train_em(problem, **kwargs):
    em = GaussianMixture(**kwargs)
    em.fit(problem.X_train)
    return em

def run_ica(problem, **kwargs):
    results = FastICA(**kwargs).fit_transform(problem.X_train)
    return results

def run_lda(problem, **kwargs):
    results = LinearDiscriminantAnalysis(**kwargs).fit_transform(problem.X_train)
    return results

def run_ica_batch(problem, **kwargs):
    results = np.zeros((ICA_ITERATIONS))
    for i in range(ICA_ITERATIONS):
        x_project, run_time = cache(run_ica, problem, random_state=RANDOM_STATE+i, **kwargs)
        num_features = x_project.shape[1]
        y = range(num_features)
        kurts_proj = np.zeros((num_features))
        for j in y:
            kurts_proj[j] = kurtosis(x_project[:, j])
        results[i] = kurts_proj.mean()
    return results

def run_rp(problem, **kwargs):
    #results = GaussianRandomProjection(**kwargs).fit_transform(problem.X_train)
    results = SparseRandomProjection(**kwargs).fit_transform(problem.X_train)
    return results


def generate_km_score_data(problem, dr_analysis, n_init, fn, **kwargs):
    scores = []
    for i in problem.k_range(dr_analysis, Analysis.KM):
        km, runtime = cache(train_kmeans, problem, n_clusters=i,
                    #init='random',
                    init='k-means++',
                    n_init=n_init,
                    max_iter=K_MEANS_MAX_ITER,
                    #tol=1e-04,
                    #random_state=RANDOM_STATE,
                    **kwargs)
        score = fn(problem, km)
        scores.append(score)
    return scores

def generate_elbow_data(problem, dr_analysis, n_init, **kwargs):
    fn = lambda problem, km: km.inertia_;
    return generate_km_score_data(problem, dr_analysis, n_init, fn, **kwargs)

def generate_silhouette_data(problem, dr_analysis, n_init, **kwargs):
    fn = lambda problem, km: silhouette_score(problem.X_train, km.labels_, metric = 'cosine')
    return generate_km_score_data(problem, dr_analysis, n_init, fn, **kwargs)

def generate_calinski_harabasz_data(problem, dr_analysis, n_init, **kwargs):
    fn = lambda problem, km: calinski_harabasz_score(problem.X_train, km.labels_)
    return generate_km_score_data(problem, dr_analysis, n_init, fn, **kwargs)

def generate_largest_cluster_data(problem, dr_analysis, n_init, **kwargs):
    fn = lambda problem, km: max(np.bincount(km.labels_)) / len(km.labels_)
    return generate_km_score_data(problem, dr_analysis, n_init, fn, **kwargs)

# Taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
def plot_silhouette(problem, dr_analysis, analysis, get_labels):
    config = problem.get_config(dr_analysis, analysis)
    if not config:
        return
    # sample_silhouette_values = silhouette_samples(problem.X_train, labels)
    X = problem.X_train
    for n_clusters in [config.best_k]: #problem.k_range(analysis):
        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        #clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        #cluster_labels = clusterer.fit_predict(X)
        cluster_labels = get_labels(problem, n_clusters)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels, metric='cosine')
        #print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        y_lower = 10
        sizes = []
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            sizes.append(size_cluster_i)
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        
        ax1.set_title("%s - %s, %s silhouette plot for k=%s" % (problem.name, dr_analysis.name, analysis.value, n_clusters))
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.gcf().set_size_inches(20, 20)
        plt.savefig('%s/silhouettes/silhouette_%s.png' % (plot_dir(problem, dr_analysis, analysis), n_clusters), bbox_inches='tight')
        plt.close()

        #print('Cluster sizes for %s are %s' % (problem.name, sizes))
        plt.hist(sizes, bins='auto')
        plt.title("%s - %s, %s bin sizes for k=%s" % (problem.name, dr_analysis.name, analysis.value, n_clusters))
        plt.grid()
        plt.xscale('log')
        plt.xticks(fontsize=12)
        plt.gcf().set_size_inches(FIG_HEIGHT, FIG_HEIGHT)
        #plt.gcf().set_size_inches(20, 20)
        plt.savefig('%s/hist/hist_%s.png' % (plot_dir(problem, dr_analysis, analysis), n_clusters),
                    bbox_inches='tight')
        plt.close()

def compute_tsne(problem, **kwargs):
    tsne = TSNE(**kwargs)
    results = tsne.fit_transform(problem.X)
    return results

# From https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
def plot_tsne(problem, dr_analysis, analysis, title, labels):
    results, runtime = cache(compute_tsne, problem, n_components=2, verbose=1,
                             perplexity=80, learning_rate=500, n_iter=300,
                             random_state=RANDOM_STATE)
    #df = pd.DataFrame.from_records(results)
    df = pd.DataFrame(columns=['First', 'Second'])
    df['First'] = results[:,0]
    df['Second'] = results[:,1]
    df['Labels'] = labels
    
    k = len(np.unique(labels))

    #plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=results[:,0], #"First",
        y=results[:,1], #y="Second",
        hue=labels, #"Labels",
        palette=sns.color_palette("hls", k),
        #data=df,
        legend=False, #"brief", # full, auto
        alpha=0.5
    )
    plt.title('%s\n%s (k=%s)' % (problem.name, title, k))
    h_score = homogeneity_score(problem.y, labels)
    c_score = completeness_score(problem.y, labels)
    ami_score = adjusted_mutual_info_score(problem.y, labels)
    if problem.name == 'Dexter' and dr_analysis == Analysis.DEFAULT:
        plt.gca().set_ylim([-60, 60])
        plt.gca().set_xlim([-60, 60])

    plt.xlabel('AMI=%.3f\nCompleteness=%.3f\nHomogeneity=%.3f' % (ami_score, c_score, h_score))
    plt.gcf().set_size_inches(FIG_HEIGHT, FIG_HEIGHT)
    plt.savefig('%s/%s_best_tsne.png' % (plot_dir(problem, dr_analysis, analysis), title), bbox_inches='tight')
    plt.close()

def get_km(problem, n_clusters):
    result, runtime = cache(train_kmeans, problem, n_clusters=n_clusters, init='k-means++',
                        n_init=K_MEANS_RESTARTS,
                        max_iter=K_MEANS_MAX_ITER,
                        random_state=RANDOM_STATE)
    return result

def get_km_labels(problem, n_clusters):
    result = get_km(problem, n_clusters)
    labels = result.labels_
    return labels    


fig_num_map = {}
curr_fig_num = 1000
def start_comparison_plot(problem, analysis, title):
    global curr_fig_num
    key = '%s - %s - %s' % (problem.name, analysis.name, title)
    if key in fig_num_map:
        fig_num = fig_num_map[key]
    else:
        fig_num_map[key] = fig_num = curr_fig_num
        curr_fig_num += 1
    plt.figure(fig_num)

scores_map = {}
def save_comparison_scores(problem, analysis, title, dr_analysis, scores):
    key = '%s - %s - %s' % (problem.name, analysis.name, title)
    
    if not key in scores_map:
        scores_map[key] = []
    scores_map[key].append((dr_analysis, scores))

def finalize_comparison_plots(problem, analysis, title):
    key = '%s - %s - %s' % (problem.name, analysis.name, title)
    #config = problem.get_config(analysis, Analysis.KM)
    plt.title('%s - %s: %s' % (problem.name, analysis.name, title))
    plt.ylabel(title)
    plt.xlabel('Number of Clusters')
    min_num_scores = 1000
    max_best_k = 0
    for dr_analysis, all_scores in scores_map[key]:
        min_num_scores = min(min_num_scores, len(all_scores))
        config = problem.get_config(dr_analysis, analysis)
        max_best_k = max(config.best_k, max_best_k)
    
    limit = max(min_num_scores, max_best_k)
        
    for dr_analysis, all_scores in scores_map[key]:
        config = problem.get_config(dr_analysis, analysis)
        x = problem.k_range(dr_analysis, analysis)[:limit]
        means = np.mean(all_scores, axis=0)[:limit]
        stds = np.std(all_scores, axis=0)[:limit]
        p = plt.plot(x, means, label='%s (k=%s)' % (dr_analysis.name, config.best_k))
        matching_indices = np.where(x == config.best_k)
        if len(matching_indices) == 0 or len(matching_indices[0]) == 0:
            print("Could not find %s in %s" % (config.best_k, x))
            continue
        index = np.where(x == config.best_k)[0][0]
        best_x = x[index]
        best_y = means[index]
        plt.plot([best_x], [best_y], marker='o', markersize=10, color=p[-1].get_color())
        plt.fill_between(x, means - stds, means + stds, alpha=0.15)

    """
    fig_num = fig_num_map[key]
    save_fig_num = plt.gcf().number
    plt.figure(fig_num)
    """
    plt.legend()
    finalize_plot(problem, analysis, 'DR comparison for %s - %s' % (analysis.name, title),
                  Analysis.DEFAULT)
    #plt.figure(save_fig_num)
    

def k_means(problem, dr_analysis):
    config = problem.get_config(dr_analysis, Analysis.KM)
    num_iterations = K_MEANS_ITERATIONS
    dims = (num_iterations, config.get_num_k())
    distortions_all_scores = np.zeros(dims)
    sil_all_scores = np.zeros(dims)
    ch_all_scores = np.zeros(dims)
    all_largest_cluster_sizes = np.zeros(dims)
    
    cache_key = {'min_k': config.min_k,
                'max_k': config.max_k};
    for i in range(0, num_iterations):
        kwargs = {
                'problem': problem,
                'dr_analysis': dr_analysis,
                'n_init': K_MEANS_RESTARTS,
                'random_state': RANDOM_STATE+i}
        distortions, dis_time = cache(generate_elbow_data, key=cache_key, **kwargs)
        distortions_all_scores[i] = np.array(distortions)
        sil_scores, sil_time = cache(generate_silhouette_data, key=cache_key, **kwargs)
        sil_all_scores[i] = np.array(sil_scores)
        ch_scores, ch_time = cache(generate_calinski_harabasz_data, key=cache_key, **kwargs)
        ch_all_scores[i] = np.array(ch_scores)
        largest_cluster_sizes, lcs_time = cache(generate_largest_cluster_data, key=cache_key, **kwargs)
        #largest_cluster_sizes = generate_largest_cluster_data(**kwargs)
        all_largest_cluster_sizes[i] = np.array(largest_cluster_sizes)
    
    x = problem.k_range(dr_analysis, Analysis.KM)
    
    #######################
    # Elbow and Silhouette
    #######################
    distortions_mean = np.mean(distortions_all_scores, axis=0)
    distortions_std = np.std(distortions_all_scores, axis=0)
    sil_mean = np.mean(sil_all_scores, axis=0)
    sil_std = np.std(sil_all_scores, axis=0)
    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_title('%s %s - Elbow and Silhouette' % (problem.name, dr_analysis.name))
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Distortion', color=color)
    ax1.plot(x, distortions_mean, linewidth=LINE_WIDTH, color=color)
    ax1.fill_between(x, distortions_mean - distortions_std,
                     distortions_mean + distortions_std,
                     alpha=0.15, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    label_best(problem, distortions_mean, dr_analysis, Analysis.KM)

    ax2 = ax1.twinx()
    color = 'tab:blue';
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.fill_between(x, sil_mean - sil_std,
                     sil_mean + sil_std,
                     alpha=0.15, color=color)
    ax2.plot(x, sil_mean, linewidth=LINE_WIDTH, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    label_best(problem, sil_mean, dr_analysis, Analysis.KM)
    finalize_plot(problem, Analysis.KM, 'elbow_silhouette', dr_analysis)
    
    ##################################
    # Calinsski Harabasz, Cluster sizes
    ##################################
    for name, all_scores in [('Calinsski Harabasz', ch_all_scores),
                             ('Largest Cluster Size % of Samples', all_largest_cluster_sizes),
                             ('Silhouette', sil_all_scores)]:
        save_comparison_scores(problem, Analysis.KM, name, dr_analysis, all_scores)
        #start_comparison_plot(problem, Analysis.KM, name)
        means = np.mean(all_scores, axis=0)
        stds = np.std(all_scores, axis=0)
        plt.plot(x, means, label=dr_analysis.name)
        plt.fill_between(x, means - stds, means + stds, alpha=0.15)
        dr_name = ''
        if dr_analysis != Analysis.DEFAULT:
            dr_name = '[%s]' % (dr_analysis.name)
        plt.title('%s - KM%s %s scores' % (problem.name, dr_name, name))
        plt.ylabel(name)
        plt.xlabel('Number of Clusters')
        label_best(problem, means, dr_analysis, Analysis.KM)
        finalize_plot(problem, Analysis.KM, name, dr_analysis)
    
    ##################################
    # Chosen K
    ##################################
    if not config.best_k:
        return None
    km, runtime = cache(get_km, problem=problem, n_clusters=config.best_k)
    labels = km.predict(problem.X)
    for scorer in [homogeneity_score, completeness_score, adjusted_mutual_info_score]:
        score = scorer(problem.y, labels)
        print("%s - %s: %s = %s" % (problem.name, Analysis.KM.name, scorer.__name__, score))
    dr_name = '[%s]' % (dr_analysis.name)
    if dr_analysis == Analysis.DEFAULT:
        dr_name = ''
    plot_tsne(problem, dr_analysis, Analysis.KM, "%s%s - Cluster labels" % (Analysis.KM.name, dr_name), labels)
    new_problem = problem.clone(labels.reshape(-1, 1))

    return new_problem, labels
    

def get_em_labels(problem, n_components):
    result, runtime = cache(train_em, problem, n_components=n_components, n_init=EM_RESTARTS,
                            max_iter=K_MEANS_MAX_ITER,
                            covariance_type='diag',
                            random_state=RANDOM_STATE)
    labels = result.predict(problem.X_train)
    return labels

# min_k, max_k just for caching
def get_em_data(problem, dr_analysis, num_iterations, num_restarts):
    config = problem.get_config(dr_analysis, Analysis.EM)
    num_k = config.get_num_k()
    aic_means = np.zeros((num_k))
    aic_stds = np.zeros((num_k))
    bic_means = np.zeros((num_k))
    bic_stds = np.zeros((num_k))
    sil_means = np.zeros((num_k))
    sil_stds = np.zeros((num_k))
    ch_means = np.zeros((num_k))
    ch_stds = np.zeros((num_k))
    db_means = np.zeros((num_k))
    db_stds = np.zeros((num_k))
    x = problem.k_range(dr_analysis, Analysis.EM)
    for index in range(len(x)):
        n_components = x[index]
        aic = np.zeros((num_iterations))
        bic = np.zeros((num_iterations))
        sil = np.zeros((num_iterations))
        ch = np.zeros((num_iterations))
        db = np.zeros((num_iterations))
        for j in range(0, num_iterations):
            result, runtime = cache(train_em, problem, n_components=n_components,
                                    #init='random',
                                    n_init=num_restarts,
                                    max_iter=K_MEANS_MAX_ITER,
                                    #tol=1e-1,
                                    covariance_type='diag',
                                    random_state=RANDOM_STATE + j)
            aic[j] = result.aic(problem.X_train)
            bic[j] = result.bic(problem.X_train)
            labels = result.predict(problem.X_train)
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances
            # TODO: try different metrics here
            sil[j] = silhouette_score(problem.X_train, labels, metric='euclidean',
                                         random_state=RANDOM_STATE + j)
            ch[j] = calinski_harabasz_score(problem.X_train, labels)
            db[j] = davies_bouldin_score(problem.X_train, labels)
        aic_means[index] = np.mean(aic)
        aic_stds[index] = np.std(aic)
        bic_means[index] = np.mean(bic)
        bic_stds[index] = np.std(bic)
        sil_means[index] = np.mean(sil)
        sil_stds[index] = np.std(sil)
        ch_means[index] = np.mean(ch)
        ch_stds[index] = np.std(ch)
        db_means[index] = np.mean(db)
        db_stds[index] = np.std(db)
    return {"aic": {"means": aic_means, "stds": aic_stds},
            "bic": {"means": bic_means, "stds": bic_stds},
            "sil": {"means": sil_means, "stds": sil_stds},
            "ch": {"means": ch_means, "stds": ch_stds},
            "db": {"means": db_means, "stds": db_stds},
            }

# Look at MDL: http://erikerlandson.github.io/blog/2016/08/03/x-medoids-using-minimum-description-length-to-identify-the-k-in-k-medoids/
# More MDL: https://machinelearningmastery.com/probabilistic-model-selection-measures/
def em(problem, dr_analysis):
    config = problem.get_config(dr_analysis, Analysis.EM)
    if not config:
        return
    x = problem.k_range(dr_analysis, Analysis.EM)
    cache_key = {"min_k": config.min_k, "max_k": config.max_k}
    data, runtime = cache(get_em_data, problem, dr_analysis=dr_analysis, num_iterations=EM_ITERATIONS,
                          num_restarts=EM_RESTARTS, key=cache_key)
    aic_means = data["aic"]["means"]
    aic_stds = data["aic"]["stds"]
    bic_means = data["bic"]["means"]
    bic_stds = data["bic"]["stds"]
    
    ###########################
    # AIC vs BIC
    ###########################
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_title('%s - Expectation Maximization' % (problem.name))
    if config.best_k:
        ax1.set_xlabel('Information Criterion (best=%s)' % (config.best_k))
    else:
        ax1.set_xlabel('Information Criterion')
    ax1.set_ylabel('AIC', color=color)
    ax1.plot(x, aic_means, linewidth=LINE_WIDTH, color=color)
    ax1.fill_between(x, aic_means - aic_stds, aic_means + aic_stds, alpha=0.15, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    label_best(problem, aic_means, dr_analysis, Analysis.EM)

    ax2 = ax1.twinx()
    color = 'tab:blue';
    ax2.set_ylabel('BIC', color=color)
    ax2.plot(x, bic_means, linewidth=LINE_WIDTH, color=color)
    ax2.fill_between(x, bic_means - bic_stds, bic_means + bic_stds, alpha=0.15, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    label_best(problem, bic_means, dr_analysis, Analysis.EM)
    finalize_plot(problem, Analysis.EM, "aic_vs_bic", dr_analysis)

    ###########################
    # Silhouette, Calinski-Harabasz, Davies-Bouldin
    ###########################
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_title('%s - Expectation Maximization' % (problem.name))
    if config.best_k:
        ax1.set_xlabel('Number of Components (best=%s)' % (config.best_k))
    else:
        ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Davies-Bouldin', color=color)
    means = data["db"]["means"]
    stds = data["db"]["stds"]
    ax1.plot(x, means, linewidth=LINE_WIDTH, color=color)
    ax1.fill_between(x, means - stds, means + stds, alpha=0.15, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    label_best(problem, means, dr_analysis, Analysis.EM)

    ax2 = ax1.twinx()
    color = 'tab:blue';
    ax2.set_ylabel('Calinski-Harabasz', color=color)
    means = data["ch"]["means"]
    stds = data["ch"]["stds"]
    ax2.plot(x, means, linewidth=LINE_WIDTH, color=color)
    ax2.fill_between(x, means - stds, means + stds, alpha=0.15, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    label_best(problem, data["ch"]["means"], dr_analysis, Analysis.EM)
    finalize_plot(problem, Analysis.EM, "ch_vs_db", dr_analysis)
    
    for metric, key in [('Silhouette', "sil"),
                           ('Calinski-Harabasz', "ch"),
                           ('Davies-Bouldin', "db")]:
        means = data[key]["means"]
        stds = data[key]["stds"]
        plt.plot(x, means)
        plt.fill_between(x, means - stds, means + stds, alpha=0.15)
        dr_name = ''
        if dr_analysis != Analysis.DEFAULT:
            dr_name = '[%s]' % (dr_analysis.name)
        plt.title('%s - EM%s %s scores' % (problem.name, dr_name, metric))
        plt.ylabel(metric)
        plt.xlabel('Number of Components')
        label_best(problem, means, dr_analysis, Analysis.EM)
        finalize_plot(problem, Analysis.EM, metric, dr_analysis)
                
    
    ##################################
    # Chosen K
    ##################################
    if not config.best_k:
        return None
    km, runtime = cache(get_km, problem=problem, n_clusters=config.best_k)
    labels = km.predict(problem.X)

    dr_name = '[%s]' % (dr_analysis.name)
    if dr_analysis == Analysis.DEFAULT:
        dr_name = ''

    plot_tsne(problem, dr_analysis, Analysis.EM, "%s%s - Cluster labels" % (Analysis.EM.name, dr_name), labels)
    new_problem = problem.clone(labels.reshape(-1, 1))

    return new_problem, labels


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

def get_validation_curve(problem, param, param_range, num_folds, scoring):
    train_scores, test_scores = validation_curve(
        estimator=clone(problem.estimator),
        X=problem.X_train,
        y=problem.y_train,
        param_name=param,
        param_range=param_range,
        scoring=scoring,
        cv=num_folds,
        n_jobs=-1,
        verbose=1)
    return train_scores, test_scores

@ignore_warnings(category=ConvergenceWarning)
def get_best_estimator(problem, num_folds, scoring, tuning_iterations):
    """
    clf = GridSearchCV(problem.estimator, problem.param_grid, cv=num_folds,
                       scoring=scoring, n_jobs=-1, verbose=2)
    """
    clf = RandomizedSearchCV(problem.estimator, problem.param_grid, cv=num_folds,
                             scoring=scoring, n_jobs=-1, n_iter=tuning_iterations,
                             random_state=RANDOM_STATE, verbose=1)
    clf.fit(problem.X_train, problem.y_train)
    #return clf.best_params_
    return clf.best_estimator_

def score_train_and_test(problem, best_params, scoring):
    estimator = clone(problem.estimator)
    estimator.set_params(**best_params)
    estimator.fit(problem.X_train, problem.y_train)
    
    y_test_predict = estimator.predict(problem.X_test)
    y_train_predict = estimator.predict(problem.X_train)

    train_score = balanced_accuracy_score(problem.y_train, y_train_predict)
    test_score = balanced_accuracy_score(problem.y_test, y_test_predict)

    return np.round(train_score, 3), np.round(test_score, 3)

#####################
# Validation curve
# From Machine Learning with Python, and Assignment 1
#####################
def plot_validation_curves(problem, dr_analysis, analysis):
    scoring = 'balanced_accuracy'
    num_folds = 40
    param_grid = problem.param_grid
    label = plot_dir(problem, dr_analysis, analysis)
    tuning_iterations = 100
    
    param_grid_key = joblib.hash(problem.param_grid)
    estimator, tuning_time = cache(get_best_estimator, problem,
                                    key=param_grid_key,
                                    num_folds=10,
                                    scoring=scoring, 
                                    tuning_iterations=tuning_iterations)
    print("Param search took %sms", tuning_time)
    problem.estimator = estimator

    num_data_points = 30
    best_params = {} # deepcopy(estimator.get_params())
    best_result = None
    for param, param_range in param_grid.items():
        if (len(param_range) > num_data_points):
            every = int(len(param_range) / num_data_points)
            param_range = param_range[::every]
            param_range.sort()
        print()
        print('Plotting validation curve for %s' % (label))
        (train_scores, test_scores), runtime = cache(get_validation_curve,
            problem,
            param=param,
            param_range=param_range,
            num_folds=num_folds,
            scoring=scoring)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        best_param_result = max(test_mean)
        idx = np.where(test_mean == best_param_result)[0][0]
        best_param_value = param_range[idx]
        if not best_result or best_param_result > best_result:
            best_result = best_param_result
            best_params[param] = best_param_value
            
        plot_range = param_range
        if is_log(param_range):
            plt.xscale('log')
        param_name = param.split('__')[-1]

        best_param_value_str = str(best_param_value)
        if isinstance(best_param_value, (np.float64, float)):
            best_param_value_str = format_number(best_param_value)

        if isinstance(param_range[0], str):
            for i in range(len(param_range)):
                if param_range[i] is None:
                    param_range[i] = 'None'
        elif param_name == 'hidden_layer_sizes':
            if best_param_value:
                best_param_value = best_param_value[0]
            plot_range = [x[0] for x in param_range]

        is_convergence = param_name in ['max_iter']
        c = 0
        if is_convergence:
            c = test_mean[-1]
            plt.axhline(y=c, linestyle='--')
            plt.title('Convergence Plot for %s' % (label))
        else:
            plt.title('Model Complexity Curve for %s %s' % (label, param_name))
        plt.plot(plot_range, train_mean,
                 color='blue', marker='o',
                 markersize=5, label='Training ' + scoring)
        plt.fill_between(plot_range, train_mean + train_std,
                         train_mean - train_std, alpha=0.15,
                         color='blue')
        plt.plot(plot_range, test_mean,
                 color='green', linestyle='--',
                 marker='s', markersize=5,
                 label='Validation ' + scoring)
        plt.fill_between(plot_range, test_mean + test_std,
                         test_mean - test_std, alpha=0.15,
                         color='green')
        if is_convergence:
            if param_name == 'max_iter':
                plt.xlabel('Number of Iterations')
            plt.ylabel('%s [convergence at %.2f]' % (scoring, c))
        else:
            plt.xlabel('%s [best=%s]' % (param_name, best_param_value_str))
            if best_param_value is not None:
                plt.axvline(x=best_param_value, linestyle='--')
            plt.ylabel(scoring)
        plt.legend()
        finalize_plot(problem, analysis, "validation_curve_%s" % (param_name), dr_analysis)
        print('Created validation curve for ', param, '=', best_param_value)
        print()
    
    print('best_result', best_result, 'best_params', best_params)
    train_result, test_result = score_train_and_test(problem, best_params, scoring)
    # TODO: Also show train result?
    return test_result, tuning_time

def visualize_dr(problem, analysis, components=[0, 1]):
    plt.scatter(problem.X_test[:, components[0]], problem.X_test[:, components[1]],
                c=problem.y_test, alpha=0.5)
    title = "%s Features %s for %s" % (analysis.name, components, problem.name)
    plt.title(title)
    #plt.xscale('log')
    #plt.yscale('log')
    finalize_plot(problem, analysis, title, width=FIG_HEIGHT)

# https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn
def reconstruction_error(problem, algorithm):
    # Taken from PCA implementation:
    # https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/decomposition/pca.py#L419
    # The mean is added back in PCA implementation, but since we pretransform
    # with standard score, the data already has mean zero.
    X_proj = algorithm.transform(problem.X_train)
    X_reconstructed = X_proj @ algorithm.components_
    error = np.mean(np.square(problem.X_train - X_reconstructed))
    return error
    #print("Error is %.6G" % error)

# https://stackoverflow.com/questions/38799205/why-are-my-manual-pca-reconstructions-not-matching-pythons-sklearns-reconstruc
# https://stackoverflow.com/questions/1730600/principal-component-analysis-in-python/12273032#12273032
# https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/decomposition/pca.py#L419
def get_reconstruction_error(problem, algorithm, analysis):
    x = problem.k_range(Analysis.DEFAULT, analysis)
    config = problem.get_config(Analysis.DEFAULT, analysis)
    errors_mean = np.zeros((config.get_num_k()))
    errors_std = np.zeros((config.get_num_k()))
    for i in range(len(x)):
        k = x[i]
        iter_errors = np.zeros((RP_ITERATIONS))
        for j in range(RP_ITERATIONS):
            rp = algorithm(random_state=RANDOM_STATE + j,
                           n_components=k).fit(problem.X_train)
            iter_errors[j] = reconstruction_error(problem, rp)
        errors_mean[i] = np.mean(iter_errors)
        errors_std[i] = np.std(iter_errors)
    return errors_mean, errors_std

def get_rp_error(problem, analysis, min_k, max_k):
    return get_reconstruction_error(problem, GaussianRandomProjection, analysis)

def get_pca_error(problem, analysis, min_k, max_k):
    return get_reconstruction_error(problem, PCA, analysis)

def rp_summary(problem):
    rp_config = problem.get_config(Analysis.DEFAULT, Analysis.RP)
    pca_config = problem.get_config(Analysis.DEFAULT, Analysis.PCA)
    if not rp_config: return
    pca_x = problem.k_range(Analysis.DEFAULT, Analysis.PCA)
    #rp_x =  problem.k_range(Analysis.RP)
    # Compute based on PCA k range
    (rp_errors_mean, rp_errors_std), rp_time = cache(get_rp_error, problem,
        analysis = Analysis.PCA, min_k=pca_config.min_k, max_k=pca_config.max_k)
    (pca_errors_mean, pca_errors_std), pca_time = cache(get_pca_error, problem,
        analysis = Analysis.PCA, min_k=pca_config.min_k, max_k=pca_config.max_k)

    ##################################
    # Plot error by component count
    ##################################
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_title('%s - Reconstruction Error' % (problem.name))
    ax1.set_ylabel('Randomized Projection Recon. Error', color=color)
    #ax1.set_yscale('log')
    ax1.plot(pca_x, rp_errors_mean, linewidth=LINE_WIDTH, color=color)
    ax1.fill_between(pca_x, rp_errors_mean - rp_errors_std,
                     rp_errors_mean + rp_errors_std,
                     alpha=0.15, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    label_best(problem, rp_errors_mean, Analysis.DEFAULT, Analysis.RP, color)

    ax2 = ax1.twinx()
    color = 'tab:blue';
    ax2.set_ylabel('PCA Recon. Error', color=color)
    ax2.plot(pca_x, pca_errors_mean, linewidth=LINE_WIDTH, color=color)
    ax2.fill_between(pca_x, pca_errors_mean - pca_errors_std,
                     pca_errors_mean + pca_errors_std,
                     alpha=0.15, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    label_best(problem, pca_errors_mean, Analysis.DEFAULT, Analysis.PCA, color)

    ax1.set_xlabel('Number of components (RP best=%s, PCA best=%s)'
                   % (format_number(rp_config.best_k), format_number(pca_config.best_k)))

    finalize_plot(problem, Analysis.RP, 'reconstruction_error')
    
    eps_range = np.linspace(0.01, 0.2, 50)
    dims = johnson_lindenstrauss_min_dim(problem.X_train.shape[0], eps=eps_range)
    plt.yscale('log')
    plt.plot(eps_range, dims, linewidth=LINE_WIDTH, markersize=LINE_WIDTH)
    plt.title("Johnson-Lindenstrauss Bound for %s" % (problem.name))
    plt.ylabel("Min # of dimensions")
    plt.xlabel("Epsilon (#samples=%s)" % (problem.X_train.shape[0]))
    finalize_plot(problem, Analysis.RP, 'johnson-lindenstrauss')
    
    rp = GaussianRandomProjection(random_state=RANDOM_STATE,
                                  n_components=rp_config.best_k)
    new_X = rp.fit_transform(problem.X)
    new_problem = problem.clone(new_X)
    visualize_dr(new_problem, Analysis.RP)
    return new_problem
    
    """
    To make this work, need to loop over RP dimensions until a good start and end range is found.
    May need to make range logarithmic.
    ##################################
    # Plot component count by error
    ##################################
    # Recompute based on RP k range
    (rp_errors_mean, rp_errors_std), rp_time = cache(get_rp_error, problem,
        analysis = Analysis.RP, min_k=rp_config.min_k, max_k=rp_config.max_k)
    num_x = min(len(pca_x), len(rp_x))

    x2 = pca_errors_mean[::-1]
    y2 = rp_x[::-1][:num_x]
    print("pca error levels", x2)
    print("rp #components", y2)
    print("rp error levels", rp_errors_mean[::-1])
    y3 = np.interp(x2, rp_errors_mean[::-1][:num_x], y2)
    print("rp #components", y3)
    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_title('%s - Components Required for Error Level' % (problem.name))
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Randomized Projection #Components', color=color)
    #ax1.set_yscale('log')
    ax1.plot(x2, y3, linewidth=LINE_WIDTH, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', which='major', labelsize=12)

    ax2 = ax1.twinx()
    color = 'tab:blue';
    ax2.set_ylabel('PCA #Components', color=color)
    ax2.plot(x2, y2, linewidth=LINE_WIDTH, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    finalize_plot(problem, pca_errors_mean, Analysis.RP, 'components_by_reconstruction_error')
    """

# From Python Machine Learning 3rd edition
def plot_cum_sum(explained_variance_ratio):
    cum_sum = np.cumsum(explained_variance_ratio)
    r = range(1, len(explained_variance_ratio))
    plt.bar(r, explained_variance_ratio, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(r, cum_sum, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()

# https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
def compute_pca_scores(problem):
    pca = PCA(random_state=RANDOM_STATE, whiten=True)

    pca_scores = []
    x = problem.k_range(Analysis.DEFAULT, Analysis.PCA)
    for k in x:
        pca.n_components = k
        pca_scores.append(np.mean(cross_val_score(pca, problem.X_train)))

    return x, pca_scores

def pca_summary(problem):
    config = problem.get_config(Analysis.DEFAULT, Analysis.PCA)
    if not config: return
    pca = PCA(random_state=RANDOM_STATE, whiten=True)
    pca.fit(problem.X_train)
    error = reconstruction_error(problem, pca)
    print("PCA reconstruction error is %.6G" % error)
    # Note that this is giving the distribution of eigenvalues
    y = pca.explained_variance_ratio_[:config.max_k]
    
    # From Python Machine Learning 3rd edition
    x = range(0, len(y))
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_title("PCA Explained Variance Ratio for %s" % (problem.name))
    ax1.set_ylabel("Explained Variance Ratio", color=color)
    ax1.plot(x, y, linewidth=LINE_WIDTH, markersize=LINE_WIDTH, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    label_best(problem, y, Analysis.DEFAULT, Analysis.PCA)
    
    ax2 = ax1.twinx()
    color = 'tab:blue';
    ax2.set_ylabel("Cumulative Explained Variance Ratio", color=color)
    ax2.bar(x, y, alpha=0.5, align='center',
            label='Individual explained variance')
    ax2.step(x, np.cumsum(y), where='mid', label='Cumulative explained variance')
    #plt.legend(loc='best')
    
    label_best(problem, np.cumsum(y), Analysis.DEFAULT, Analysis.PCA)
    finalize_plot(problem, Analysis.PCA, 'explained_variance_ratio')
    
    (x2, pca_scores), time = cache(compute_pca_scores, problem, key="v2")
    plt.title("PCA Cross Validation Score %s" % (problem.name))
    plt.plot(x2, pca_scores, linewidth=LINE_WIDTH, markersize=LINE_WIDTH)
    plt.ylabel("CV Score")
    plt.xlabel("Number of Components")
    label_best(problem, pca_scores, Analysis.DEFAULT, Analysis.PCA)
    finalize_plot(problem, Analysis.PCA, 'cross_val_score')
    
    rp = PCA(random_state=RANDOM_STATE, whiten=True, n_components=config.best_k)
    new_X = rp.fit_transform(problem.X)
    new_problem = problem.clone(new_X)
    visualize_dr(new_problem, Analysis.PCA)
    return new_problem

def lda_summary(problem):
    config = problem.get_config(Analysis.DEFAULT, Analysis.LDA)
    if not config: return
    lda = LinearDiscriminantAnalysis()
    lda.fit(problem.X_train, problem.y_train)
    # Note that this is giving the distribution of eigenvalues
    y = lda.explained_variance_ratio_[:config.max_k]
    
    # From Python Machine Learning 3rd edition
    x = range(0, len(y))
    #plt.plot(x, y, linewidth=LINE_WIDTH, markersize=LINE_WIDTH)
    plt.bar(x, y, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(x, np.cumsum(y), where='mid', label='Cumulative explained variance')
    plt.title("PCA Explained Variance Ratio for %s" % (problem.name))
    plt.ylabel("Explained Variance Ratio")
    plt.xlabel("Principal Component Index")
    plt.legend(loc='best')
    
    #label_best(problem, y, Analysis.DEFAULT, Analysis.LDA)
    finalize_plot(problem, Analysis.LDA, 'explained_variance_ratio')
    
    rp = LinearDiscriminantAnalysis(n_components=config.best_k)
    new_X = rp.fit_transform(problem.X, problem.y)
    new_problem = problem.clone(new_X)
    #visualize_dr(new_problem, Analysis.LDA)
    return new_problem

def get_ica_data(problem, min_k, max_k):
    config = problem.get_config(Analysis.DEFAULT, Analysis.ICA)
    if not config: return
    x = problem.k_range(Analysis.DEFAULT, Analysis.ICA)
    num_k = len(x)
    avg_proj_kurts = np.zeros((num_k))
    std_proj_kurts = np.zeros((num_k))
    for i in range(len(x)):
        k = x[i]
        kurts_proj, run_time = cache(run_ica_batch, problem, n_components=k, whiten=True)
        avg_proj_kurts[i] = kurts_proj.mean()
        std_proj_kurts[i] = kurts_proj.std()
    
    kurts = np.zeros((problem.X_train.shape[1]))
    for i in range(problem.X_train.shape[1]):
        kurts[i] = kurtosis(problem.X_train[:, i])
    avg_kurt = kurts.mean()

    return avg_proj_kurts, std_proj_kurts, avg_kurt

# Kurtosis tells us how gaussian our components are. We want to find the components
# that are the most independent. We want the place where the components are the
# most non-gaussian (independent). In PCA, if we add an additional component (n), then
# components 1 through n-1 stay the same. In ICA, if we add an additional component,
# then the entire solution can change. Some people select the best number of
# components based on the average kurtosis.
# When kurtosis is high, our components are less Gaussian, which is what we want.
# Why use kurtosis, and maybe use tanh instead: https://towardsdatascience.com/separating-mixed-signals-with-independent-component-analysis-38205188f2f4
def ica_summary(problem):
    config = problem.get_config(Analysis.DEFAULT, Analysis.ICA)
    if not config: return
    x = problem.k_range(Analysis.DEFAULT, Analysis.ICA)
    (avg_proj_kurts, std_proj_kurts, avg_kurt), runtime = cache(get_ica_data, problem, min_k=config.min_k, max_k=config.max_k)
    
    plt.plot(x, avg_proj_kurts, linewidth=LINE_WIDTH, markersize=LINE_WIDTH)
    plt.fill_between(x, avg_proj_kurts - std_proj_kurts,
                     avg_proj_kurts + std_proj_kurts,
                     alpha=0.15)
    plt.title("ICA Kurtosis for %s" % (problem.name))
    plt.xlabel("Num Features")
    plt.ylabel("Avg Kurtosis")
    #plt.axhline(y=avg_kurt, linestyle='--')
    label_best(problem, avg_proj_kurts, Analysis.DEFAULT, Analysis.ICA)
    finalize_plot(problem, Analysis.ICA, 'kurtosis')

    rp = FastICA(random_state=RANDOM_STATE, n_components=config.best_k)
    new_X = rp.fit_transform(problem.X)
    new_problem = problem.clone(new_X)
    kurts_proj = []
    for j in range(new_X.shape[1]):
        kurts_proj.append(kurtosis(new_X[:, j]))
    z = zip(kurts_proj, np.arange(len(kurts_proj)).tolist())
    components = [c for _, c in sorted(z, reverse=True)]
    visualize_dr(new_problem, Analysis.ICA, components=components)
    return new_problem


def decision_tree(problem):
    config = problem.get_config(Analysis.DEFAULT, Analysis.DT)
    if not config: return
    # TODO: Maybe random forest would be a better choice?
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(problem.X_train, problem.y_train)
    # From Python Machine Learning 3rd edition
    importances = dt.feature_importances_
    indices = np.argsort(importances)[::-1]
    indices = indices[importances[indices] > 0.005]
    labels = ["X%05d" % (i) for i in indices]
    x = range(len(indices))
    for f in x:
        print("%2d) %-*s %f" % (f + 1, 10, labels[f], importances[indices[f]]))
    plt.title('%s (%s feature importances > 0.005)' % (problem.name, len(indices)))
    plt.ylabel("Feature/Gini importance")
    plt.bar(x, importances[indices], align='center')
    plt.xticks(x, labels, rotation=90)
    plt.xlim([-1, len(x)])
    label_best(problem, importances[indices], Analysis.DEFAULT, Analysis.DT)
    finalize_plot(problem, Analysis.DT, 'feature_importances')
    
    new_X = problem.X[:, indices]
    new_problem = problem.clone(new_X)
    visualize_dr(new_problem, Analysis.DT)
    return new_problem

def mkdirs(dr_analysis, analysis, problem):
    base_dir = plot_dir(problem, dr_analysis, analysis)
    dirs = [base_dir]
    if analysis in [Analysis.KM, Analysis.EM]:
        dirs.append('%s/silhouettes' % (base_dir))
        dirs.append('%s/hist' % (base_dir))
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name);

nn_results = {}
cluster_results = {}
cluster_sizes = {}

def cluster_experiments(problem, dr_analysis):
    for analysis in [Analysis.DEFAULT, Analysis.KM, Analysis.EM]:
        mkdirs(dr_analysis, analysis, problem)
    km_results, km_labels = k_means(problem, dr_analysis)
    plot_tsne(problem, dr_analysis, Analysis.DEFAULT, "Ground truth", problem.y)
    plot_silhouette(problem, dr_analysis, Analysis.KM, get_km_labels)
    plot_silhouette(problem, dr_analysis, Analysis.EM, get_em_labels)
    em_results, em_labels = em(problem, dr_analysis)
    
    km_ami_score = adjusted_mutual_info_score(problem.y, km_labels)
    km_clusters = problem.get_config(dr_analysis, Analysis.KM).best_k
    em_ami_score = adjusted_mutual_info_score(problem.y, em_labels)
    em_clusters = problem.get_config(dr_analysis, Analysis.EM).best_k
    dr_name = dr_analysis.name + ' - '
    if dr_analysis == Analysis.DEFAULT:
        dr_name = ''
    cluster_results[problem.name][dr_name + Analysis.KM.name] = km_ami_score
    cluster_sizes[problem.name][dr_name + Analysis.KM.name] = km_clusters
    cluster_results[problem.name][dr_name + Analysis.EM.name] = em_ami_score
    cluster_sizes[problem.name][dr_name + Analysis.EM.name] = em_clusters
    
    if dr_analysis == Analysis.DEFAULT:
        nn_results[problem.name][Analysis.KM.name] = \
            plot_validation_curves(km_results, dr_analysis, Analysis.KM)
        nn_results[problem.name][Analysis.EM.name] = \
            plot_validation_curves(em_results, dr_analysis, Analysis.EM)

# TODO: Plot AMI for chosen K for KM, EM
"""
Instructor: For the initial portion of the assignment, you derive the clusters in an
unsupervised manner; you shouldn't be using the ground truth labels or any
supervised methods here. After you choose K, you may validate your clusters
against the ground truth labels at that point.
"""

# Based on https://stackoverflow.com/a/50354131
def label_bars(text_format, scale='linear', **kwargs):
    ax = plt.gca()
    bars = ax.patches

    if scale == 'log':
        heights = [math.log(bar.get_height()) for bar in bars]
    else:
        heights = [bar.get_height() for bar in bars]
    max_height = max(heights)

    for bar in bars:
        text = text_format % (bar.get_height())
        color = 'white'
        va = 'center'
        ha = 'center'
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + bar.get_height() / 2
        if scale == 'log' and math.log(bar.get_height()) / max_height < 0.2:
            color = 'black'
            va = 'bottom'
            text_y = (bar.get_y() + bar.get_height()) * 1.1 #0.2 * max_height)

        ax.text(text_x, text_y, text, ha=ha, va=va, color=color, **kwargs)

def plot_nn_results(problem_results, time_scale):
    titles = ['Balanced Accuracy', 'Tuning Time (s)']
    reverse = [True, False]
    scales = ['linear', time_scale]
    for i in range(2):
        results = []
        for label, score in problem_results.items():
            results.append((score[i], label))
        scores, labels = [ list(p) for p in zip(*sorted(results, reverse=reverse[i]))]
        print("labels", labels)
        print("scores", scores)
        plt.title('%s - NN results' % (problem.name))
        plt.ylabel(titles[i])
        x = range(len(labels))
        plt.bar(x, scores, align='center')
        plt.xticks(x, labels, rotation=90)
        plt.xlim([-1, len(x)])
        plt.yscale(scales[i])
        label_bars('%.2f', scales[i], rotation=90)
        finalize_plot(problem, Analysis.DEFAULT, 'nn_results_%s' % (titles[i]))

def plot_cluster_results(problem_results, cluster_sizes):
    results = []
    for label, score in problem_results.items():
        results.append((score, cluster_sizes[label], label))
    scores, sizes, labels = [ list(p) for p in zip(*sorted(results, reverse=True))]
    print("labels", labels)
    print("scores", scores)
    print("sizes", sizes)
    plt.title('%s - Clustering results' % (problem.name))
    plt.ylabel("AMI score")
    x = range(len(labels))
    plt.bar(x, scores, align='center')
    plt.xticks(x, labels, rotation=90)
    plt.xlim([-1, len(x)])
    label_bars('%.2f', rotation=90)

    ax = plt.gca()
    bars = ax.patches
    for i in range(len(bars)):
        bar = bars[i]
        text = 'k=%d' % (sizes[i])
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + bar.get_height() + max(scores) * 0.03
        ax.text(text_x, text_y, text, ha='center',
                va='bottom', color='black')

    plt.ylim([0, max(scores) * 1.2])
    finalize_plot(problem, Analysis.DEFAULT, 'cluster_results_ami')
    
# TODO: Summarized results for DR/clustering combinations. Use AMI.
if __name__ == "__main__":
    print("CWD: " + os.getcwd())
    if not os.path.exists('cache'):
        os.makedirs('cache')
    
    for problem_fn in [polish_bankruptcy, dexter, dexter_like_noise]:
        problem = Problem.new(problem_fn)
        for analysis in Analysis:
            mkdirs(Analysis.DEFAULT, analysis, problem)
        if problem_fn == dexter_like_noise:
            k_means(problem, Analysis.DEFAULT)
            continue
        nn_results[problem.name] = {}
        cluster_results[problem.name] = {}
        cluster_sizes[problem.name] = {}
        cluster_experiments(problem, Analysis.DEFAULT)

        pca_results = pca_summary(problem)
        ica_results = ica_summary(problem)
        rp_results = rp_summary(problem)
        dt_results = decision_tree(problem)
        lda_results = lda_summary(problem)
        for analysis, results in [(Analysis.PCA, pca_results),
                        (Analysis.ICA, ica_results),
                        (Analysis.RP, rp_results),
                        (Analysis.DT, dt_results)]:
            
            nn_results[problem.name][analysis.name] = \
                plot_validation_curves(results, Analysis.DEFAULT, analysis)
            print('$$$$$$$$$$$$$$$$$$$$$$$$')
            print('$$$$$$$$ %s %s' % (problem.name, analysis.name))
            print('$$$$$$$$$$$$$$$$$$$$$$$$')
            cluster_experiments(results, analysis)

        finalize_comparison_plots(problem, Analysis.KM, 'Calinsski Harabasz')
        finalize_comparison_plots(problem, Analysis.KM, 'Largest Cluster Size % of Samples')
        finalize_comparison_plots(problem, Analysis.KM, 'Silhouette')
        
        nn_results[problem.name]['Default'] = \
            plot_validation_curves(problem, Analysis.DEFAULT, Analysis.DEFAULT)
    
        time_scale = 'linear'
        if problem.name == 'Dexter':
            time_scale = 'log'
        plot_nn_results(nn_results[problem.name], time_scale)
        plot_cluster_results(cluster_results[problem.name], cluster_sizes[problem.name])
        continue
        
    
    print()
    print('Final results:')
    print(nn_results)
    
    os.system("./glue_plots.sh")

"""
Office Hours 9:

Silhouette score doesn't penalize complexity?

Only training should be used for clustering experiments.
Hold out set should only be used for NN.

Choosing between metrics that disagree is difficult, should choose metric that
best fits data.

PCA - explained variance
RP - reconstruction error
ICA - kurtosis or simple classifier to see which solution loses the least amount of info

For 16 combinations, don't dive deeply into all of them, just the more interesting ones.
But *do* give the results of all 16 combinations.

For step 5, clustering is on original data sets. Don't need to include DR, just
clustering and then NN.

***TODO***: Visualize DR, 22min
May be difficult to visualize DR, just choose first two or three components.
Scatterplot on first two (or three) components. TSNE may be possible here.

Using labels for 4th DR algo is fine.

***TODO***: Use one-hot encoding for cluster labels.
Use one-hot encoding to do NN on cluster labels.

# Office hour 9, 38min.
Don't want to use same data for clustering and NN, so need to divide data into
three parts. It's fine to use the transformation determined on the training set
and apply to the test set.

If there are not independent components, then ICA won't give great results.

If DR gives same balanced accuracy, probably not losing any info from DR.
"""