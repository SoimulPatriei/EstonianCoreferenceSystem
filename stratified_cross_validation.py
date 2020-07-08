#!/usr/bin/env python

"""This module is not a part of the Estonian coreference system but an auxiliary module"""
"""It applies stratified cross validation for the selection of the most performat classifier to be used by the testing module"""


___author__ = "Eduard Barbu"
__license__ = "LGPL"
__version__ = "1.0.0"
__maintainer__ = "Eduard Barbu"
__email__ = "barbu@ut.ee"

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import classifiers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import logging
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
import time


"""Ignore warnings python -W ignore stratified_cross_validation.py"""

def init_logger(logging_file):
    """Init the logger for the console and logging file"""

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=logging_file,
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format

    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)



def get_feature_type (f_feature_names) :
    """Get the type of features in a dict"""

    dict_feature_type ={}

    fi = open(f_feature_names, mode='r', encoding='utf-8')
    for line in fi :
        line = line.rstrip()
        f_name, f_type = line.split("\t")
        dict_feature_type[f_name]=f_type
    fi.close()

    return dict_feature_type


def oversample_function (X_train, y_train) :

    logger = logging.getLogger("obersampling_adasyn")

    ada = ADASYN()
    logger.info("ADASYN()")
    X_ros, y_ros = ada.fit_sample(X_train, y_train)
    return X_ros, y_ros


def undersample_function (X_train,y_train) :

    logger = logging.getLogger("random_undersampling")
    logger.info ("random_state=0, sampling_strategy=0.5")
    rus = RandomUnderSampler(random_state=0, sampling_strategy=0.5)
    X_rus, y_rus = rus.fit_sample(X_train, y_train)
    return X_rus, y_rus


def identical_function (X_train,y_train) :

    return X_train,y_train


def classify (X,y,model,transform_function) :
    """Classify based on any trained model using Stratified K folds"""

    list_f1=[]
    n_folds=4
    skf=StratifiedKFold(n_splits=n_folds, random_state=1, shuffle=True)
    for train_index, test_index in skf.split(X,y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index],y[train_index], y[test_index]
        X_train,y_train = transform_function (X_train,y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        list_f1.append(f1)
    return np.mean(list_f1)



def getXy (f_model,f_feature_names) :
  """Get X and y"""

  df = pd.read_csv(f_model)
  features = df.columns

  #of course the last feature is the category
  X = df[features[:-1]]

  dict_feature_type = get_feature_type(f_feature_names)
  categorical_features = [feature for feature in dict_feature_type if dict_feature_type[feature] == 'categorical']
  ct = ColumnTransformer(
      [('one_hot_encoder', OneHotEncoder(categories='auto'), categorical_features)],
      remainder='passthrough'
  )
  X_ohe = ct.fit_transform(X)

  y = np.array(df['category'])

  return X_ohe,y


def logistic_regression (X,y) :
    logger = logging.getLogger("balanced_classification logistic_regression")

    logger.info("weight classification")
    start_time = time.time()
    model, message = classifiers.get_logistic_regression_classifier_2 ()
    f1_measure=classify(X, y, model,identical_function)
    logger.info(message +" {}".format(round(f1_measure,2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))

    start_time = time.time()
    model, message = classifiers.get_logistic_regression_classifier_1()
    f1_measure = classify(X, y, model,undersample_function)
    logger.info("undersample " +message + " {}".format(round(f1_measure, 2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))

    start_time = time.time()
    model, message = classifiers.get_logistic_regression_classifier_1()
    f1_measure = classify(X, y, model,oversample_function)
    logger.info("oversample " + message + " {}".format(round(f1_measure, 2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))



def xgboost (X,y) :

    logger = logging.getLogger("balanced_classification")

    logger.info("weight classification")
    start_time = time.time()
    model, message = classifiers.get_xgboost_classifier ()
    f1_measure=classify(X, y, model,identical_function)
    logger.info(message +" {}".format(round(f1_measure,2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))

    start_time = time.time()
    model, message = classifiers.get_xgboost_classifier ()
    f1_measure = classify(X, y, model,undersample_function)
    logger.info("undersample "+message + " {}".format(round(f1_measure, 2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))

    start_time = time.time()
    model, message = classifiers.get_xgboost_classifier ()
    f1_measure = classify(X, y, model,oversample_function)
    logger.info("oversample " + message + " {}".format(round(f1_measure, 2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))


def gradient_boost (X,y) :

    logger = logging.getLogger("balanced_classification")

    logger.info("weight classification")
    start_time = time.time()
    model, message = classifiers.get_gradient_boosting_classfier ()
    f1_measure=classify(X, y, model,identical_function)
    logger.info(message +" {}".format(round(f1_measure,2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))

    start_time = time.time()
    model, message = classifiers.get_gradient_boosting_classfier()
    f1_measure = classify(X, y, model,undersample_function)
    logger.info("undersample "+message + " {}".format(round(f1_measure, 2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))

    start_time = time.time()
    model, message = classifiers.get_gradient_boosting_classfier ()
    f1_measure = classify(X, y, model,oversample_function)
    logger.info("oversample " + message + " {}".format(round(f1_measure, 2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))


def balanced_classification (X,y) :

    """Classify using algorithms with different techniques for balancing the data"""

    logistic_regression(X, y)
    gradient_boost(X, y)
    xgboost(X, y)


def imbalanced_classification (X,y) :

    """Classify using the algorithms (with the original class distribution) """

    logger = logging.getLogger("imbalanced_classification")

    start_time = time.time()
    model, message = classifiers.get_gradient_boosting_classfier()
    f1_measure = classify(X, y, model, identical_function)
    logger.info(message + " {}".format(round(f1_measure, 2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))

    start_time = time.time()
    model, message = classifiers.get_xgboost_classifier ()
    f1_measure = classify(X, y, model, identical_function)
    logger.info(message + " {}".format(round(f1_measure, 2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))

    start_time = time.time()
    model,message=classifiers.get_decision_tree_classifier()
    f1_measure=classify(X, y, model,identical_function)
    logger.info(message +" {}".format(round(f1_measure,2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))

    start_time = time.time()
    model, message = classifiers.get_k_nearst_neighbors ()
    f1_measure = classify(X, y, model,identical_function)
    logger.info(message +" {}".format(round(f1_measure, 2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))

    start_time = time.time()
    model, message = classifiers.get_logistic_regression_classifier_1 ()
    f1_measure=classify(X, y, model,identical_function)
    logger.info(message +" {}".format(round(f1_measure,2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))

    start_time = time.time()
    model, message = classifiers.get_dummy_classifier ()
    f1_measure=classify(X, y, model,identical_function)
    logger.info(message +" {}".format(round(f1_measure,2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))


def compute_proportions(fPath) :
    """Glance the data: show it is imbalanced, compute class proportions"""

    logger = logging.getLogger("compute_proportions")
    df = pd.read_csv(fPath)

    category_count = df.category.value_counts()
    logger.info('Class 0: {}'.format (category_count[0]))
    logger.info('Class 1: {}'.format(category_count[1]))
    class_proportion = round(category_count[0] / category_count[1],2)

    logger.info('Proportion 0 to 1 : {}'.format(class_proportion))
    return category_count


def main():

    f_model = "estonian_results/estonian_training_corpus-sklearn.txt"
    f_feature_names = "estonian_results/estonian-computed-features.txt"

    logging_file = "estonian_results/stratified_cross_validation_info.txt"
    init_logger(logging_file)
    logger = logging.getLogger("stratified_cross_validation")
    start_time = time.time()
    compute_proportions(f_model)
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))


    logger.info ("Read and normalize X and y")
    start_time = time.time()
    X,y=getXy(f_model, f_feature_names)
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))
    imbalanced_classification(X, y)
    balanced_classification(X, y)




if __name__ == '__main__':
  main()

