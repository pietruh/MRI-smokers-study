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
    get_t_test_results, feature_selector_simple, run_loocv, Kernel_SVM, explore_selection_methods, build_results_comparison

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import LeaveOneOut
from sklearn.utils import shuffle
import os


#%matplotlib inline
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
#boxplot_features(data)
#normalize_volume_as_proportion(data)


# Normalizing volume as a proportion to total volume for each sex
data_norm = normalize_data_sex(data)

data_shuffled0 = shuffle(data_norm, random_state=7)
data_shuffled0 = data_shuffled0.reset_index(drop=True)

y0 = data_shuffled0.grupa
X0 = data_shuffled0.drop(['grupa'], axis=1)

smoker = data_shuffled0[data_shuffled0['grupa']== 1]
nsmoker = data_shuffled0[data_shuffled0['grupa']== 0]


# Define models (these could be models that have interpretable features)
# Returns vector of acc results with increasing number of features
logreg = LogisticRegression(solver='lbfgs')
# Build a forest and compute the feature importances
forest = RandomForestClassifier(n_estimators=100,
                              random_state=0)

# Define feature selectors
t_test = get_t_test_results(data_shuffled0, smoker, nsmoker)

feature_scores = mutual_info_classif(X0, y0)
inf_gain = pd.DataFrame({'variable': list(X0), 'score': -np.sort(-feature_scores)})

model = forest.fit(X0, y0)
importances = model.feature_importances_
rf_imp = pd.DataFrame({'variable': list(X0), 'score': -np.sort(-importances)})

# Get accuracy of feature selection for logreg and different selection methods
feature_selectors_list = [t_test, inf_gain, rf_imp]
for i, c in enumerate(feature_selectors_list):
    scores_big_array_np = feature_selector_simple(X0, y0, feature_selectors_list[i], logreg, test_size=0.2, num_iters=1, len_plot=10)


# TODO: continue refactoring from below
# call_PCA_processing(data_shuffled0, X0) # FIXME: Use this if in need


data_shuffled = shuffle(data_norm)
data_shuffled = data_shuffled.reset_index(drop=True)
y = data_shuffled.grupa
X = data_shuffled.drop(['grupa'], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)

X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

X_np = np.array(X)
y_np = np.array(y)

#explore_selection_methods(X, X_val, y, y_val)  # FIXME: svm is crashing here


# Function for running the whole classification process, after applying feature selection ,
# and with varying validation set ratios
last_to_feature_list = 15
inf_gain_list = inf_gain['variable'][1:last_to_feature_list].tolist()
t_test_list = t_test['variable'][1:last_to_feature_list].tolist()
rf_imp_list = rf_imp['variable'][1:last_to_feature_list].tolist()
#fs_list = [rf_imp_list, t_test_list, inf_gain_list]
fs_list = [rf_imp, t_test, inf_gain]

#   arr_acc, arr_prec, arr_clf = build_results_comparison(fs_list, data_norm)  # FIXME: For now there is no grid search
#print(f"arr_acc = {arr_acc}")
#print(f"arr_prec = {arr_prec}")

# Running 3x3 classification, with different feature selectors and different classifiers
## TODO: Remember to pass models after grid search (the ones with the parameters that gives the best results )
#

random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
svmc = svm.SVC()
lr = LogisticRegression(solver='lbfgs')
clasf = [random_forest, svmc, lr]
clasf_names =['random_forest', 'svmc', 'lr']
fs_names =  ['rf_imp', 't_test', 'inf_gain']
#arr_clf_flatten = arr_clf.flatten()
#classifiers = [arr_clf_flatten[i].best_estimators_ for i in arr_clf_flatten]
clf_list = []
description = []
num_iters = 100
fig = plt.figure(figsize=(10, 8))
plt.xlabel('#features', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.title('Plot of different models and feature selectors', fontsize=20)
linestyles = ['-', ':', '--', '-.']     # linestyle for fs
model_colors_dict =['k', 'r', 'b'] # color for classifiers#{'random_forest': 'k', 'svmc': 'r', 'lr': 'b'}
len_plot = 30
for j, f in enumerate(fs_list):
    for i, c in enumerate(clasf):
        #print("interesting part is here", i, c, j, f)
        #arr2[i,j,k] = run_classification(data_norm, c, f, val_ratio = v)

        # acc_with_features[i,j] = feature_selector(X0, y0, f, c, test_size=0.2, num_iters=1000)
        tmp_res = feature_selector_simple(X, y, f, c, test_size=0.2, num_iters=num_iters, X_val=X_val, y_val=y_val, len_plot = len_plot)
        clf_list.append(tmp_res)
        description.append({'fs': f, 'c': c})
        plt.plot(tmp_res.mean(axis=0), label=clasf_names[i] + ', ' + fs_names[j], color=model_colors_dict[i],
                 linestyle=linestyles[j])

legend = fig.legend(loc='upper right', shadow=True, fontsize='large')
fig.savefig('img_comparison.png')


acc_with_features = np.asarray(clf_list).mean(axis=1)
np.save('clf_list.npy', clf_list)
# get final results to the array (should be 4d)
#big_arr_to_graph = np.array(acc_with_features)
# now, let's get mean of the last iteration dimension


#def feature_selector(X0, x_val, y0, y_val, test_method, classifier_method, test_size=0.2, num_iters=1000):
#tmp_res = feature_selector(X, X_val, y, y_val, f, c, test_size=0.2, num_iters=10)
#X0 = X
#x_val=X_val
#y0=y
#test_method=fs_list[0]
#classifier_method = classifiers[0]
#num_iters=10
#test_size=0.2
#scores_big_array = []
#n_iters = 0
#i = 0
##for n_iters in range(0, num_iters):
#    #X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=test_size)#, random_state=1)
##
##     scores = []
##  #   for i in range(1, len(test_method)):
##         classifier_method.fit(X0[test_method[0:i]['variable']], y)
##         score = classifier_method.score(x_val[test_method[0:i]['variable']], y_val)
##         scores.append(score)
##     scores_big_array.append(scores)
## scores_big_array_np = np.array(scores_big_array)
##
##
##     classifier_method.fit(X0[test_method[0:i]['variable']], y)
#
#test_method[0:10]
#
#
#