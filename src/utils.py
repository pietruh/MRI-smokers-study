"""This file contains all utility functions that does not fit well exploration in jupyter, e.g. they are too long,
 too mundane, but never pretty enough"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, GridSearchCV
from sklearn.decomposition import PCA, KernelPCA

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.utils.fixes import signature
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def preprocess_statistics(all_stats, data, remove_strange=0):
    """ Here is a simple preprocessing to remove strange data, drop meaningles columns and filtering different
    types of metrics variables
    :rtype: dataframe
    :param all_stats: pandas dataframe read from file with all of the statistics
    :param data: pandas dataframe read from fully merged data
    :return:
    """
    # Find ID of strange data (smoker with 0 years of smoking)
    strange = all_stats.loc[(all_stats['group'] == 'non') | all_stats['YearsOfSmoking'] == 0]
    strange_id = strange.Variables.str.extract('(\d+)')
    strange_id = strange_id[0].tolist()
    # data= df.iloc[:,[0,40]]
    # data = df[list(df.columns[0:1]) + list(df.columns[40:])]
    drop_id = [130, 361, 379, 382, 455, 508, 517, 566, 643, 801, 834, 854, 927, 513] + remove_strange * strange_id
    print(f'Dropping smokers with ids {drop_id}')
    data = data[~data.id.isin(drop_id)]

    data = data.replace({'grupa': {'smoker': 1, 'non-smoker': 0}})
    # Dropping meaningless columns with NaNs
    data = data.drop(['recon_all2', 'recon_all222', 'recon_all2222', 'recon_all22222', 'recon_all3'], axis=1)
    data = data.drop(['id', 'recon_all', 'r_insula_MeanCurv', 'r_temporalpole_MeanCurv', 'r_temporalpole_GrayVol',
                      'r_insula_GrayVol', 'r_insula_ThickAvg', 'r_temporalpole_ThickAvg'], axis=1)
    # filtering values based on regexes
    data_mm3 = data.filter(regex='mm3', axis=1)
    data_vol = data.filter(regex='_GrayVol', axis=1)
    data_thick = data.filter(regex='_ThickAvg', axis=1)
    data_curv = data.filter(regex='_MeanCurv', axis=1)
    # data_surfa = data.filter(regex='SurfArea', axis=1)    # FIXME: Maybe use this features in future
    data_left_right_overall = data[data.columns[0:10]]
    grupa = data[['grupa']]
    sex = data[['sex']]

    # Concatenate data and return
    data = pd.concat([grupa, sex, data_mm3, data_vol, data_thick, data_left_right_overall, data_curv], axis=1,
                     join_axes=[data_mm3.index])
    return data


def boxplot_features(data):
    """Draw boxplots of a different features (see below)"""
    brain_vol = data.boxplot(column=' Brain Segmentation Volume (mm3)', by='grupa')

    grey_vol = data.boxplot(column=' Total gray matter volume (mm3)', by='grupa')

    grey_cortical_vol = data.boxplot(column=' Total cortical gray matter volume (mm3)', by='grupa')
    return


def normalize_volume_as_proportion(data):
    """Normalizing volume as a proportion to total volume"""
    volume = data.filter(regex='_mm3', axis=1)
    data[list(volume)].div(data[' Brain Segmentation Volume (mm3)'], axis=0)
    gray_volume = data.filter(regex='_GrayVol', axis=1)
    data[list(gray_volume)].div(data[' Total cortical gray matter volume (mm3)'], axis=0)
    data['Brain_White_Surface_Total_Area'] = data[' R Cortex White Surface Total Area'] + data[
        ' L Cortex White Surface Total Area']
    surface_area = data.filter(regex='_SurfArea', axis=1)
    data[list(surface_area)].div(data['Brain_White_Surface_Total_Area'], axis=0)
    return


def normalize_data_sex(data):
    """Normalizing volume based on sex"""
    ## Divide on two dataframes
    data_woman = data.loc[data['sex'] == "k"]
    data_woman = data_woman.drop("sex", axis=1)
    data_man = data.loc[data['sex'] == "m"]
    data_man = data_man.drop("sex", axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    # Normalizing volume as a proportion to total volume

    # Normalization column-wise
    data_man_norm = pd.DataFrame(min_max_scaler.fit_transform(data_man), columns=data_man.columns)
    data_woman_norm = pd.DataFrame(min_max_scaler.fit_transform(data_woman), columns=data_woman.columns)
    data_norm = pd.concat([data_man_norm, data_woman_norm], axis=0, ignore_index=True)

    return data_norm


def get_t_test_results(data_shuffled0, smoker, nsmoker):
    t_stats = []
    p_value = []

    for i in list(data_shuffled0)[3:]:      # FIXME: it was 250 on the right bound
        test = ttest_ind(smoker[i], nsmoker[i])
        t_stats.append(test[0])
        p_value.append(test[1])

    # data frame with variables and scores, sorted
    t_test = pd.DataFrame({'variable': list(data_shuffled0)[3:], 't_statistic': t_stats, 'p_value': p_value})
    t_test = t_test.sort_values(by=['p_value'])
    return t_test


def feature_selector_simple(X0, y0, test_method, classifier_method, test_size=0.2, num_iters=1000, len_plot=None, X_val=None, y_val=None):
    """Simpler API. Without using validation set. Useful for plotting. Function that will plot how adding next features influence classifier score
    :param test_method: given order of features
    :param classifier_method: classifier"""

    len_plot = len(test_method) if len_plot is None else len_plot       # how many features to take into accountin the plot&calculations

    scores_big_array = np.zeros((num_iters, len_plot))
    for n_iters in range(0, num_iters):
        if X_val is None:
            X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=test_size)#, random_state=1)
        else:
            X_train = X0
            X_test = X_val
            y_train = y0
            y_test = y_val


        for i in range(1, len_plot):
            print(n_iters, i)
            classifier_method.fit(X_train[test_method[0:i]['variable']], y_train)
            score = classifier_method.score(X_test[test_method[0:i]['variable']], y_test)
            scores_big_array[n_iters, i] = score
    #scores_big_array_np = np.array(scores_big_array)

    # plot mean values
    plt.plot(range(1, len_plot), scores_big_array.mean(axis=0)[1:], color="black")
    plt.title("Accuracy of LogReg with t-test feature selection")
    plt.xlabel("Number Of variables")
    plt.ylabel("Accuracy Score")
    print("Mean accuracy values for different number of features ={}".format(scores_big_array.mean(axis=0)))
    return scores_big_array


def call_PCA_processing(data_shuffled0, X0):
    """Processing data after running PCA - NOT TESTED"""
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X0)

    pca.explained_variance_ratio_

    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    X_kpca = kpca.fit_transform(X0)
    X_back = kpca.inverse_transform(X_kpca)

    kpca_sig = KernelPCA(kernel="sigmoid", fit_inverse_transform=True, gamma=10)

    kpca_poly = KernelPCA(kernel="poly", degree=3, fit_inverse_transform=True, gamma=10)

    def Kernel_Pca(ker):
        kpca = KernelPCA(n_components=4, kernel=ker, gamma=15)
        x_kpca = kpca.fit_transform(X0)
        kpca_transform = kpca.fit_transform(X0)
        explained_variance = np.var(kpca_transform, axis=0)
        ev = explained_variance / np.sum(explained_variance)

        # --------- Bar Graph for Explained Variance Ratio ------------
        plt.bar([1, 2, 3, 4], list(ev * 100), label='Principal Components', color='b')
        plt.legend()
        plt.xlabel('Principal Components ')
        # ----------------------
        n = list(ev * 100)
        pc = []
        for i in range(len(n)):
            n[i] = round(n[i], 4)
            pc.append('PC-' + str(i + 1) + '(' + str(n[i]) + ')')

        # ----------------------
        plt.xticks([1, 2, 3, 4], pc, fontsize=7, rotation=30)
        plt.ylabel('Variance Ratio')
        plt.title('Variance Ratio of MRI data using kernel:' + str(ker))
        plt.show()
        # ---------------------------------------------------
        # *Since the initial 2 principal components have high variance.
        #   so, we select pc-1 and pc-2.
        # ---------------------------------------------------
        kpca = KernelPCA(n_components=2, kernel=ker, gamma=15)
        x_kpca = kpca.fit_transform(X0)
        principalComponents = kpca.fit_transform(X0)
        # QWQdeeqREdqarde
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['PC-1', 'PC-2'])
        # Adding lables
        finalDf = pd.concat([principalDf, data_shuffled0[['grupa']]], axis=1)
        # Plotting pc1 & pc2
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('PC-1', fontsize=15)
        ax.set_ylabel('PC-2', fontsize=15)
        ax.set_title('KPCA on MRI data using kernel:' + str(ker), fontsize=20)
        targets = [1, 0]
        colors = ['r', 'g']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['grupa'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'PC-1']
                       , finalDf.loc[indicesToKeep, 'PC-2']
                       , c=color
                       , s=30)
        ax.legend(targets)
        ax.grid()
        plt.show()  # FOR SHOWING THE PLOT
        # ------------------- SAVING DATA INTO CSV FILE ------------
        # finalDf.to_csv('iris_after_KPCA_using_'+str(ker)+'.csv')

    # ------------------------------------------------------
    k = ['linear', 'rbf', 'poly', 'sigmoid']
    for i in k:
        Kernel_Pca(i)


def run_k_fold(classifier, X, y, k=10):
    """Running k fold classification"""
    kf = KFold(n_splits=k, random_state=7)
    kf.get_n_splits(X)

    accuracy_set = []
    roc_score_set = []
    # precision_set =[]
    yscore_np = -np.ones(shape=y.shape)
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)

        # predictions = rfc.predict(X_test)
        score = classifier.score(X_test, y_test)
        accuracy_set.append(score)

        if isinstance(classifier.best_estimator_, svm.SVC):
            print(classifier.decision_function(X_test))
            yscore_np[test_index] = classifier.decision_function(X_test)
        else:
            # print(classifier.predict_proba(X_test))
            yscore_np[test_index] = classifier.predict_proba(X_test)[:, 0]
        # precision, recall, _ = precision_recall_curve(y_test, y_score)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, classifier.predict(X_test))
        # print (auc(false_positive_rate, true_positive_rate))
        rscore = auc(false_positive_rate, true_positive_rate)
        roc_score_set.append(rscore)

    # print (roc_auc_score(y, rfc.predict(X)))
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

    avg_acc = sum(accuracy_set) / len(accuracy_set)
    avg_auc = sum(roc_score_set) / len(roc_score_set)
    print("Average accuracy score score %f" % avg_acc)
    print("Average ROC AUC score %f" % avg_auc)
    return avg_acc, avg_auc, average_precision


def run_loocv(classifier, X, y):
    loo = LeaveOneOut()
    tot_acc = []
    for train_index, test_index in loo.split(X):
        # print("train:", train_index, "validation:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)

        score = classifier.score(X_test, y_test)
        tot_acc.append(score)
    print(sum(tot_acc) / len(tot_acc))


def Kernel_SVM(ker, X, y, k):
    ker_svm = svm.NuSVC(kernel=ker, gamma=15)
    print(ker, k)
    run_k_fold(ker_svm, X, y, k)


def explore_selection_methods(X, X_val, y, y_val):
    """ Here I am running different classifiers with different model selection techniques i.e. LOOCV/kfold"""
    # Running logistic regression with different validation setups: loocv/k fold
    logreg = LogisticRegression(solver='lbfgs')
    run_loocv(logreg, X, y)
    run_k_fold(logreg, X, y, 3)
    run_k_fold(logreg, X, y, 5)
    run_k_fold(logreg, X, y, 10)

    # Running SVM with different validation setups: loocv/k fold
    svmc = svm.SVC()
    run_loocv(svmc, X, y)
    run_k_fold(svmc, X, y, 3)
    run_k_fold(svmc, X, y, 5)
    run_k_fold(svmc, X, y, 10)

    # Running SVM with different kernels
    k = ['linear', 'poly', 'rbf', 'sigmoid']
    for i in k:
        Kernel_SVM(i, X, y, 3)
        Kernel_SVM(i, X, y, 3)
        Kernel_SVM(i, X, y, 3)

    # Running Random Forest with different kernels
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    run_loocv(rfc)
    run_k_fold(rfc, X, y, 3)
    run_k_fold(rfc, X, y, 5)
    run_k_fold(rfc, X, y, 10)

    # Testing on validation set
    print(f"Accuracy score for validation set (logistic regression) {logreg.score(X_val, y_val)}")
    print(f"Accuracy score for validation set (SVM) {svmc.score(X_val, y_val)}")
    print(f"Accuracy score for validation set  (RF) {rfc.score(X_val, y_val)}")
    return


def run_classification(data, classifier, feature_selector_list, k=10, val_ratio=0.2, nr_of_feat=20):
    # apply feature selection criteria with specified number of features
    data_reduced = data[['grupa'] + feature_selector_list[1:nr_of_feat]]

    # shuffle data
    data_shuffled = shuffle(data_reduced, random_state=7)
    data_shuffled = data_shuffled.reset_index(drop=True)
    y = data_shuffled.grupa
    X = data_shuffled.drop(['grupa'], axis=1)

    # split data into train+test and validation sets
    X, X_val, y, y_val = train_test_split(X, y, test_size=val_ratio, random_state=1)
    X = np.array(X)
    y = np.array(y)

    # grid search for best model meta parameters for the classifier
    # FIXME: For fast run commented out grid search parameter packs
    param_rf = {'bootstrap': [True], 'n_estimators': [100]}#{'bootstrap': [True, False], 'n_estimators': [50, 100, 150]}
    param_svm = {'kernel': ['rbf'], 'C': [10]}  #{'kernel': ['rbf'], 'C': [1, 10, 100]}
    param_lr = {'C': [10]}  #{'C': [1, 10, 100]}
    #param_xgb = {'xgb.XGBClassifier': [5, 10, 25], 'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.1]}


    if isinstance(classifier, RandomForestClassifier):
        clf = GridSearchCV(classifier, param_rf, cv=5)
    if isinstance(classifier, svm.SVC):
        clf = GridSearchCV(classifier, param_svm, cv=5)
    if isinstance(classifier, LogisticRegression):
        clf = GridSearchCV(classifier, param_lr, cv=5)
    # if classifier == xgbc:
    #    clf = GridSearchCV(classifier, param_xgb, cv=5)
    clf_grid = clf.fit(X, y)

    # run model classification
    avg_acc, avg_roc, average_precision = run_k_fold(clf_grid, X, y, k)

    # calculate validation accuracy (accuracy of trained model on the data initially left out)
    val_acc = clf_grid.score(X_val, y_val)

    # return average accuracy on test sets, average auc score on test sets, and validation score
    return val_acc, average_precision, clf  # avg_roc, val_acc
    # print (avg_acc, avg_roc, val_acc)


def build_results_comparison(fs_list, data_norm):
    # Feature selection feature lists:
    # Classifiers:
    random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
    svmc = svm.SVC()
    lr = LogisticRegression(solver='lbfgs')
    clasf = [random_forest, svmc, lr]

    # Validation set ratio
    val_set = [0.2]#[0.1, 0.2, 0.25]

    # # number of features
    # feat_nr = [5, 10, 15, 20]
    #
    # # k-fold
    # k_fold = [3, 5, 10]


    # run through all combinations in a loop (feature selection + classifier)
    # features set at 20, k fold at 10
    arr_acc = np.array([[[0.0 for i in val_set] for j in fs_list] for k in clasf])
    arr_prec = np.array([[[0.0 for i in val_set] for j in fs_list] for k in clasf])
    list_clf = []

    for i, c in enumerate(clasf):
        for j, f in enumerate(fs_list):
            for k, v in enumerate(val_set):
                print(i, c, j, f, k, v)
                arr_acc[i, j, k], arr_prec[i, j, k], clf = run_classification(data_norm, c, f, k=10, val_ratio=v, nr_of_feat=20)
                list_clf.append(clf)
    arr_clf = np.asarray(list_clf).reshape((len(clasf), len(fs_list), len(val_set)))
    return arr_acc, arr_prec, arr_clf



