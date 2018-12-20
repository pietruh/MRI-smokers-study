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
    get_t_test_results, feature_selector_simple

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
    scores_big_array_np = feature_selector_simple(X0, y0, feature_selectors_list[i], logreg, test_size=0.2, num_iters=50)


# TODO: continue refactoring from below

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X0)

pca.explained_variance_ratio_

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X0)
X_back = kpca.inverse_transform(X_kpca)

kpca_sig=KernelPCA(kernel="sigmoid", fit_inverse_transform=True, gamma=10)

kpca_poly=KernelPCA(kernel="poly", degree = 3, fit_inverse_transform=True, gamma=10)

def Kernel_Pca(ker):
    kpca = KernelPCA(n_components=4, kernel=ker, gamma=15)
    x_kpca = kpca.fit_transform(X0)
    kpca_transform = kpca.fit_transform(X0)
    explained_variance = np.var(kpca_transform, axis=0)
    ev = explained_variance / np.sum(explained_variance)

    #--------- Bar Graph for Explained Variance Ratio ------------
    plt.bar([1,2,3,4],list(ev*100),label='Principal Components',color='b')
    plt.legend()
    plt.xlabel('Principal Components ')
    #----------------------
    n=list(ev*100)
    pc=[]
    for i in range(len(n)):
            n[i]=round(n[i],4)
            pc.append('PC-'+str(i+1)+'('+str(n[i])+')')

    #----------------------
    plt.xticks([1,2,3,4],pc, fontsize=7, rotation=30)
    plt.ylabel('Variance Ratio')
    plt.title('Variance Ratio of MRI data using kernel:'+str(ker))
    plt.show()
    #---------------------------------------------------
    # *Since the initial 2 principal components have high variance.
    #   so, we select pc-1 and pc-2.
    #---------------------------------------------------
    kpca = KernelPCA(n_components=2, kernel=ker, gamma=15)
    x_kpca = kpca.fit_transform(X0)
    principalComponents = kpca.fit_transform(X0)
#QWQdeeqREdqarde
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['PC-1', 'PC-2'])
    # Adding lables
    finalDf = pd.concat([principalDf, data_shuffled0[['grupa']]], axis = 1)
    # Plotting pc1 & pc2
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC-1', fontsize = 15)
    ax.set_ylabel('PC-2', fontsize = 15)
    ax.set_title('KPCA on MRI data using kernel:'+str(ker), fontsize = 20)
    targets = [1, 0]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['grupa'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'PC-1']
                   , finalDf.loc[indicesToKeep, 'PC-2']
                   , c = color
                   , s = 30)
    ax.legend(targets)
    ax.grid()
    plt.show() # FOR SHOWING THE PLOT
    #------------------- SAVING DATA INTO CSV FILE ------------
    #finalDf.to_csv('iris_after_KPCA_using_'+str(ker)+'.csv')


#------------------------------------------------------
k=['linear','rbf','poly','sigmoid' ]
for i in k:
    Kernel_Pca(i)

#Right hemisphere cortical gray matter volume (mm3)
#Total cortical gray matter volume (mm3)
#Mask Volume (mm3)
#Left hemisphere cortical gray matter volume (mm3)
#Total gray matter volume (mm3)
#Supratentorial volume (mm3)
# Brain Segmentation Volume Without Ventricles (mm3)
# Estimated Total Intracranial Volume (mm3)
                                             
data_red= data_norm
#data_red = data[['grupa', ' Right hemisphere cortical gray matter volume (mm3)', ' Total cortical gray matter volume (mm3)', ' Mask Volume (mm3)', ' Left hemisphere cortical gray matter volume (mm3)', ' Total gray matter volume (mm3)', ' Supratentorial volume (mm3)', ' Brain Segmentation Volume Without Ventricles (mm3)', ' Estimated Total Intracranial Volume (mm3)']]                                             

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.utils.fixes import signature

def run_k_fold(classifier, X, y, k = 10):
    kf = KFold (n_splits = k,random_state=7)
    kf.get_n_splits(X)
    
    accuracy_set=[]
    roc_score_set=[]
    #precision_set =[]
    yscore_np = -np.ones(shape=y.shape)
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)

        #predictions = rfc.predict(X_test)
        score = classifier.score(X_test, y_test)
        accuracy_set.append(score)
        
        if classifier == svmc:
            print(classifier.decision_function(X_test))
            yscore_np[test_index] = classifier.decision_function(X_test)
        else:
            #print(classifier.predict_proba(X_test))
            yscore_np[test_index] = classifier.predict_proba(X_test)[:,0]
        #precision, recall, _ = precision_recall_curve(y_test, y_score)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, classifier.predict(X_test))
        #print (auc(false_positive_rate, true_positive_rate))
        rscore = auc(false_positive_rate, true_positive_rate)
        roc_score_set.append(rscore)

    #print (roc_auc_score(y, rfc.predict(X)))
    average_precision = average_precision_score(y, yscore_np)

    print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
    
    precision, recall, _ = precision_recall_curve(y, yscore_np)
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
    
    
    avg_acc = sum(accuracy_set)/len(accuracy_set)
    avg_auc = sum(roc_score_set)/len(roc_score_set)
    print ("Average accuracy score score %f" % avg_acc)
    print ("Average ROC AUC score %f" % avg_auc)
    return avg_acc, avg_auc, average_precision

def run_loocv(classifier):
    loo = LeaveOneOut()
    tot_acc=[]
    for train_index, test_index in loo.split(X):
            #print("train:", train_index, "validation:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            classifier.fit(X_train, y_train)

            score = logreg.score(X_test, y_test)
            tot_acc.append(score)
    print(sum(tot_acc)/len(tot_acc))

from sklearn.model_selection import LeaveOneOut
from sklearn.utils import shuffle

data_shuffled = shuffle(data_red)
data_shuffled = data_shuffled.reset_index(drop=True)
y = data_shuffled.grupa
X = data_shuffled.drop(['grupa'], axis=1)

X= pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)

X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

X=np.array(X)
y=np.array(y)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = 'lbfgs')

run_loocv(logreg)

run_k_fold(logreg, X, y, 3)
run_k_fold(logreg, X, y, 5)
run_k_fold(logreg, X, y, 10)

from sklearn import svm
svmc = svm.SVC()

run_loocv(svmc)

run_k_fold(svmc,X, y, 3)
run_k_fold(svmc, X, y, 5)
run_k_fold(svmc,X, y, 10)

def Kernel_SVM(ker, X, y, k):
    ker_svm = svm.NuSVC(kernel=ker, gamma=15)
    print (ker, k)
    run_k_fold(ker_svm, X, y, k)


k=['linear', 'poly', 'rbf', 'sigmoid']
for i in k:
    Kernel_SVM(i, X, y, 3)
    Kernel_SVM(i, X, y, 3)
    Kernel_SVM(i, X, y, 3)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=0)


run_loocv(rfc)

run_k_fold(rfc, X, y, 3)
run_k_fold(rfc, X, y, 5)
run_k_fold(rfc, X, y, 10)

from xgboost import XGBClassifier

#model = XGBClassifier()
#model.fit(X, y)
#print(model)

#run_k_fold(model, 3)

# feature importance
#print(model.feature_importances_)
# plot
#import matplotlib.pyplot as plt
#plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
#plt.show()

# Testing on validation set

print("Accuracy score score for validation set (logistic regression) %f" % (logreg.score(X_val, y_val)))

print("Accuracy score score for validation set (SVM) %f" % (svmc.score(X_val, y_val)))

print("Accuracy score score for validation set  (RF) %f" % (rfc.score(X_val, y_val)))

# Funtion for running the whole calssification process, after applying feature selection , 
# and with varying validation set ratios
from sklearn.model_selection import GridSearchCV

def run_classification( data, classifier, feature_selector_list, k = 10, val_ratio = 0.2, nr_of_feat = 20):
    # apply feature selection criteria with specified number of features
    data_reduced= data[['grupa'] + feature_selector_list[1:nr_of_feat]]
    
    
    # shuffle data 
    data_shuffled = shuffle(data_reduced, random_state=7)
    data_shuffled = data_shuffled.reset_index(drop=True)
    y = data_shuffled.grupa
    X = data_shuffled.drop(['grupa'], axis=1)

    
    # split data into train+test and validation sets
    X, X_val, y, y_val = train_test_split(X, y, test_size=val_ratio, random_state=1)
    X=np.array(X)
    y=np.array(y)
    
    #grid search for best model meta parameters for the classifier 
    param_rf = {'bootstrap': [True, False],
                'n_estimators': [50, 100, 150]}
    param_svm = {'kernel': ['rbf'], 'C':[1, 10, 100]}
    param_lr = {'C':[1, 10, 100]}
    param_xgb = {'xgb.XGBClassifier': [5, 10, 25], 'n_estimators': [50,100,150], 'learning_rate': [0.05, 0.1]}
    
    if classifier == random_forest:
        clf = GridSearchCV(classifier, param_rf, cv=5)
    if classifier == svmm:
        clf = GridSearchCV(classifier, param_svm, cv=5)
    if classifier == lr:
        clf = GridSearchCV(classifier, param_lr, cv=5)
    #if classifier == xgbc:
    #    clf = GridSearchCV(classifier, param_xgb, cv=5)
    clf_grid = clf.fit(X, y)
    
    # run model classification
    avg_acc, avg_roc, average_precision = run_k_fold(clf_grid, X, y)
    
    # calculate validation accuracy (accuracy of trained model on the data initially left out)
    val_acc = clf_grid.score(X_val, y_val)
    
    #return average accuracy on test sets, average auc score on test sets, and validation score
    return val_acc, average_precision # avg_roc, val_acc
    #print (avg_acc, avg_roc, val_acc)

# Feature selection feature lists:
fs_list = [rf_imp_list, t_test_list, inf_gain_list]

# Classifiers:
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
svmm = svm.SVC()
lr = LogisticRegression(solver = 'lbfgs')
#xgbc = XGBClassifier()

clasf = [random_forest, svmc, lr ]#,xgbc ]

# Validation set ratio
val_set = [0.1, 0.2, 0.25]

# number of features
feat_nr = [5, 10, 15, 20]

# k-fold
k_fold = [3,5,10]

#run_classification( data_norm, random_forest, 10, rf_imp_list, val_ratio = 0.2, nr_of_feat = 20)

# run through all combinations i a loop (feature selection + classifier)
# features set at 20, k fold at 10
arr_acc = np.array([[[0.0 for i in val_set] for j in fs_list] for k in clasf])
arr_prec = np.array([[[0.0 for i in val_set] for j in fs_list] for k in clasf])
 
for i,c in enumerate(clasf):
    for j,f in enumerate(fs_list):
        for k,v in enumerate(val_set):
            print( i,c,j,f,k,v)
            arr_acc[i,j,k], arr_prec[i,j,k] = run_classification(data_norm, c, f, val_ratio = v)
            
            

print(arr)

arr2 = np.array([[[0.0 for i in val_set] for j in fs_list] for k in clasf])
 
for i,c in enumerate(clasf):
    for j,f in enumerate(fs_list):
        for k,v in enumerate(val_set):
            print( i,c,j,f,k,v)
            arr2[i,j,k] = run_classification(data_norm, c, f, val_ratio = v)

arr2


# Running 3x3 classification, with different feature selectors and different classifiers
# TODO: Remember to pass models after grid search (the ones with the parameters that gives the best results )
#run_classification( data_norm, random_forest, 10, rf_imp_list, val_ratio = 0.2, nr_of_feat = 20)

#grid search for best model meta parameters for the classifier 

def gridsearch_fit(classifier, X, y):
    param_rf = {'bootstrap': [True, False],
            'n_estimators': [50, 100, 150]}
    param_svm = {'kernel': ['rbf'], 'C':[1, 10, 100]}
    param_lr = {'C':[1, 10, 100]}
    param_xgb = {'xgb.XGBClassifier': [5, 10, 25], 'n_estimators': [50,100,150], 'learning_rate': [0.05, 0.1]}
    
    if classifier == rf_clf:
        clf = GridSearchCV(classifier, param_rf, cv=5)
    if classifier == svm_clf:
        clf = GridSearchCV(classifier, param_svm, cv=5)
    if classifier == lr_clf:
        clf = GridSearchCV(classifier, param_lr, cv=5)
    #if classifier == xgbc:
    #    clf = GridSearchCV(classifier, param_xgb, cv=5)
    clf_grid = clf.fit(X, y)
    return clf_grid

# split into set for train/test and independent validation set

#data_reduced= data[['grupa'] + feature_selector_list[1:nr_of_feat]]


# shuffle data 
data_shuffled = shuffle(data_norm, random_state=11)
data_shuffled = data_shuffled.reset_index(drop=True)
y = data_shuffled.grupa
X = data_shuffled.drop(['grupa'], axis=1)


# split data into train+test and validation sets
X, X_val, y, y_val = train_test_split(X, y, test_size=.2, random_state=1)


rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)
svm_clf = svm.SVC()
lr_clf = LogisticRegression(solver = 'lbfgs')

classifiers = [rf_clf, svm_clf, lr_clf ]

for c in classifiers:
    c = gridsearch_fit(c, X, y)

classifiers

fs_list = [rf_imp_list, t_test_list, inf_gain_list]

acc_with_features = np.array([[0.0 for j in fs_list] for k in classifiers])
acc_with_features_list = []  #np.array([[0.0 for j in fs_list] for k in clasf])
#def feature_selector(X0, x_val, y0, y_val, test_method, classifier_method, test_size=0.2, num_iters=1000):

clf_list = []
for i,c in enumerate(classifiers):
    for j,f in enumerate(fs_list):
        print( i,c,j,f)
        #arr2[i,j,k] = run_classification(data_norm, c, f, val_ratio = v)

        # acc_with_features[i,j] = feature_selector(X0, y0, f, c, test_size=0.2, num_iters=1000)
        tmp_res = feature_selector(X, X_val, y, y_val, f, c, test_size=0.2, num_iters=10)
        clf_list.append(tmp_res)
    acc_with_features_list.append(clf_list)
# get final results to the array (should be 4d)
big_arr_to_graph = np.array(acc_with_features_list)
# now, let's get mean of the last dimension


#def feature_selector(X0, x_val, y0, y_val, test_method, classifier_method, test_size=0.2, num_iters=1000):
#tmp_res = feature_selector(X, X_val, y, y_val, f, c, test_size=0.2, num_iters=10)
X0 = X
x_val=X_val
y0=y
test_method=fs_list[0]
classifier_method = classifiers[0]
num_iters=10
test_size=0.2
scores_big_array = []
n_iters = 0
i = 0
#for n_iters in range(0, num_iters):
    #X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=test_size)#, random_state=1)
#
#     scores = []
#  #   for i in range(1, len(test_method)):
#         classifier_method.fit(X0[test_method[0:i]['variable']], y)
#         score = classifier_method.score(x_val[test_method[0:i]['variable']], y_val)
#         scores.append(score)
#     scores_big_array.append(scores)
# scores_big_array_np = np.array(scores_big_array)
#
#
#     classifier_method.fit(X0[test_method[0:i]['variable']], y)

test_method[0:10]


