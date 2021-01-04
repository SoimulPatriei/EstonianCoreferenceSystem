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
from sklearn.metrics import precision_score
import logging
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
import time
import argparse
import sys


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
    list_precision=[]
    n_folds=4
    skf=StratifiedKFold(n_splits=n_folds, random_state=1, shuffle=True)
    for train_index, test_index in skf.split(X,y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index],y[train_index], y[test_index]
        X_train,y_train = transform_function (X_train,y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        precision=precision_score(y_test,y_pred)

        list_f1.append(f1)
        list_precision.append(precision)

    return [np.mean(list_f1), np.mean(list_precision)]



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
    f1_measure, precision=classify(X, y, model,identical_function)
    print_results(precision, f1_measure, logger, message, start_time)

    start_time = time.time()
    model, message = classifiers.get_logistic_regression_classifier_1()
    f1_measure, precision = classify(X, y, model,undersample_function)
    print_results(precision, f1_measure, logger, "undersample "+message, start_time)


    start_time = time.time()
    model, message = classifiers.get_logistic_regression_classifier_1()
    f1_measure, precision = classify(X, y, model,oversample_function)
    print_results(precision, f1_measure, logger, "oversample"+message, start_time)



def xgboost (X,y) :

    logger = logging.getLogger("balanced_classification")

    logger.info("weight classification")
    start_time = time.time()
    model, message = classifiers.get_xgboost_classifier ()
    f1_measure,precision=classify(X, y, model,identical_function)
    print_results(precision, f1_measure, logger, message, start_time)

    start_time = time.time()
    model, message = classifiers.get_xgboost_classifier ()
    f1_measure, precision = classify(X, y, model,undersample_function)
    print_results(precision, f1_measure, logger, message, start_time)

    start_time = time.time()
    model, message = classifiers.get_xgboost_classifier ()
    f1_measure, precision = classify(X, y, model,oversample_function)
    print_results(precision, f1_measure, logger, message, start_time)


def gradient_boost (X,y) :

    logger = logging.getLogger("balanced_classification")

    logger.info("weight classification")
    start_time = time.time()
    model, message = classifiers.get_gradient_boosting_classfier ()
    f1_measure,precision=classify(X, y, model,identical_function)
    print_results(precision, f1_measure, logger, message, start_time)

    start_time = time.time()
    model, message = classifiers.get_gradient_boosting_classfier()
    f1_measure, precision = classify(X, y, model,undersample_function)
    print_results(precision, f1_measure, logger, message, start_time)

    start_time = time.time()
    model, message = classifiers.get_gradient_boosting_classfier ()
    f1_measure, precision = classify(X, y, model,oversample_function)
    print_results(precision, f1_measure, logger, message, start_time)


def balanced_classification (X,y) :

    """Classify using algorithms with different techniques for balancing the data"""

    logistic_regression(X, y)
    gradient_boost(X, y)
    xgboost(X, y)



def print_results (precision,f1_measure,logger, message,start_time) :
    """Print the results (f1_score and precision) in a nice format"""

    logger.info("\n")
    logger.info(message)
    logger.info("precision" + " {}".format(round(precision, 2)))
    logger.info("f1_measure" + " {}".format(round(f1_measure, 2)))
    logger.info("--- %s seconds ---" % (round(time.time() - start_time)))

def imbalanced_classification (X,y) :

    """Classify using the algorithms (with the original class distribution) """

    logger = logging.getLogger("imbalanced_classification")

    start_time = time.time()
    model, message = classifiers.get_gradient_boosting_classfier()
    f1_measure, precision = classify(X, y, model, identical_function)
    print_results(precision, f1_measure, logger, message, start_time)


    start_time = time.time()
    model, message = classifiers.get_xgboost_classifier ()
    f1_measure,precision = classify(X, y, model, identical_function)
    print_results (precision,f1_measure,logger, message,start_time)

    start_time = time.time()
    model,message=classifiers.get_decision_tree_classifier()
    f1_measure,precision=classify(X, y, model,identical_function)
    print_results (precision,f1_measure,logger, message,start_time)

    start_time = time.time()
    model, message = classifiers.get_k_nearst_neighbors ()
    f1_measure,precision = classify(X, y, model,identical_function)
    print_results (precision,f1_measure,logger, message,start_time)

    start_time = time.time()
    model, message = classifiers.get_logistic_regression_classifier_1 ()
    f1_measure,precision=classify(X, y, model,identical_function)
    print_results (precision,f1_measure,logger, message,start_time)

    start_time = time.time()
    model, message = classifiers.get_dummy_classifier ()
    f1_measure,precision=classify(X, y, model,identical_function)
    print_results (precision,f1_measure,logger, message,start_time)


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
    parser = argparse.ArgumentParser(
        description='Stratified Cross Validation')
    parser.add_argument("--model", type=str, help="The relative path to the trained model")
    parser.add_argument("--feature_names_file", type=str, help="The file containing the names of the computed features")
    parser.add_argument("--logging_file", type=str, help="The file where I log the results")

    args = parser.parse_args()

    if not (args.model and args.feature_names_file and args.logging_file):
        logging.error("Wrong arguments!")
        sys.exit()

    f_model = args.model
    f_feature_names = args.feature_names_file
    logging_file = args.logging_file


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

