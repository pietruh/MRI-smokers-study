"""This file contains all utility functions that does not fit well exploration in jupyter, e.g. they are too long,
 too mundane, but never pretty enough"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    t_test = pd.DataFrame({'variable': list(data_shuffled0)[3:250], 't_statistic': t_stats, 'p_value': p_value})
    t_test = t_test.sort_values(by=['p_value'])
    return t_test


def feature_selector_simple(X0, y0, test_method, classifier_method, test_size=0.2, num_iters=1000):
    """Simpler API. Without using validation set. Useful for plotting. Function that will plot how adding next features influence classifier score
    :param test_method: given order of features
    :param classifier_method: classifier"""

    scores_big_array = []
    for n_iters in range(0, num_iters):
        X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=test_size)#, random_state=1)

        scores = []
        for i in range(1, len(test_method)):
            classifier_method.fit(X_train[test_method[0:i]['variable']], y_train)
            score = classifier_method.score(X_test[test_method[0:i]['variable']], y_test)
            scores.append(score)
        scores_big_array.append(scores)
    scores_big_array_np = np.array(scores_big_array)

    # plot mean values
    plt.plot(range(1, 223), scores_big_array_np.mean(axis=0), color="black")
    plt.title("Accuracy of LogReg with t-test feature selection")
    plt.xlabel("Number Of variables")
    plt.ylabel("Accuracy Score")
    print("Mean accuracy values for different number of features ={}".format(scores_big_array_np.mean(axis=0)))
    return scores_big_array_np
