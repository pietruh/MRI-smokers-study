"""Main script for processing and classifying smoker data"""
import pandas as pd
import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_ind

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA, KernelPCA

from utils import preprocess_statistics, boxplot_features, normalize_volume_as_proportion, normalize_data_sex, \
    get_t_test_results, feature_selector_simple, run_loocv, Kernel_SVM, explore_selection_methods, build_results_comparison,\
    run_classification, grid_search_with_feature_selectors, iterate_and_plot

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import LeaveOneOut
from sklearn.utils import shuffle
import os


# %matplotlib inline
df_path = '../data/fully_merged_data.xlsx'
try:
    df = pd.read_excel(df_path)
except:
    raise FileNotFoundError

all_stats_path = '../data/all_stats_eng.xlsx'
try:
    all_stats = pd.read_excel(all_stats_path)
except:
    raise FileNotFoundError

data = preprocess_statistics(all_stats, df, remove_strange=0)

# FIXME: Use this if in need
# boxplot_features(data)
# normalize_volume_as_proportion(data)


# Normalizing volume as a proportion to total volume for each sex
data_norm = normalize_data_sex(data)

data_shuffled0 = shuffle(data_norm, random_state=7)
data_shuffled0 = data_shuffled0.reset_index(drop=True)

y0 = data_shuffled0.grupa.astype('int')
X0 = data_shuffled0.drop(['grupa'], axis=1)

smoker = data_shuffled0[data_shuffled0['grupa']== 1]
nsmoker = data_shuffled0[data_shuffled0['grupa']== 0]
X, X_val, y, y_val = train_test_split(X0, y0, test_size=0.2, random_state=1)


# Define models (these could be models that have interpretable features)
# Returns vector of acc results with increasing number of features
logreg = LogisticRegression(solver='lbfgs')
logreg = LogisticRegression(solver='newton-cg', C=0.7, dual=False)
svm_params = {'C': 0.5,
              'tol': 1e-9,
              'cache_size': 1000,
              'kernel': 'rbf',
              'gamma': 'auto',
              'class_weight': 'balanced'}
# Build a forest and compute the feature importances
forest = RandomForestClassifier(n_estimators=100,
                              random_state=0)

# Define feature selectors
t_test = get_t_test_results(data_shuffled0, smoker, nsmoker)

feature_scores = mutual_info_classif(X0, y0)
inf_gain = pd.DataFrame({'variable': list(X0), 'score': -feature_scores}).sort_values('score')

model = forest.fit(X0, y0)
importances = model.feature_importances_
rf_imp = pd.DataFrame({'variable': list(X0), 'score': -importances}).sort_values('score')

# Get accuracy of feature selection for logreg and different selection methods
feature_selectors_list = [t_test, inf_gain, rf_imp]
rfc_ = RandomForestClassifier(n_estimators=100, random_state=0)
# svm_ = clf = svm.SVC(C=100.0, tol=1e-2, cache_size=1000, kernel='rbf', gamma=0.03,
#               class_weight='balanced')
svm_ = svm.SVC(C=400.0, tol=1e-1, cache_size=1000, kernel='rbf', gamma=0.05,
              class_weight='balanced')
svm_ = svm.SVC(C=100.0, tol=1e-1, cache_size=1000, kernel='rbf', gamma=0.03,
              class_weight='sigmoid')
svm_ = svm.SVC(**svm_params)
#for i, c in enumerate(feature_selectors_list):
#    scores_big_array_np = feature_selector_simple(X, y, feature_selectors_list[i], svm_, test_size=0.2, num_iters=5,
#    len_plot=300, X_val=X_val, y_val=y_val)


# TODO: continue refactoring from below
# call_PCA_processing(data_shuffled0, X0) # FIXME: Use this if in need


# explore_selection_methods(X, X_val, y, y_val)  # FIXME: svm is crashing here


# Function for running the whole classification process, after applying feature selection ,
# and with varying validation set ratios


fs_list = [rf_imp, t_test, inf_gain]

#   arr_acc, arr_prec, arr_clf = build_results_comparison(fs_list, data_norm)  # FIXME: For now there is no grid search
# print(f"arr_acc = {arr_acc}")
# print(f"arr_prec = {arr_prec}")

# Look for the best parameters of a classifiers
best_clfs = grid_search_with_feature_selectors(X, X_val, y, y_val, feature_selectors_list)

# Running 3x3 classification, with different feature selectors and different classifiers
# TODO: Remember to pass models after grid search (the ones with the parameters that gives the best results)
iterate_and_plot(fs_list, best_clfs)

# now, let's get mean of the last iteration dimension

