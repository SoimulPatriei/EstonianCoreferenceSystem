
""""Different classifiers from scikit-learn are returned"""

__author__ = "Eduard Barbu"
__license__ = "LGPL"
__version__ = "1.0.0"
__maintainer__ = "Eduard Barbu"
__email__ = "barbu@ut.ee"

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


def get_xgboost_classifier () :

    model = XGBClassifier()
    message = "XGBoost \tParameters(default)"
    return model, message


def get_gradient_boosting_classfier () :

   model = GradientBoostingClassifier(random_state=0)
   message = "Gradient Boosting Classifier\tParameters(random_state=0)"
   return model, message

def get_svm_linear_kernel () :

    model = SVC(kernel='linear')
    message = "SVM\tParameters(kernel='linear')"
    return model, message

def get_k_nearst_neighbors ():

    model = KNeighborsClassifier(n_neighbors=3)
    message="k Nearst Neighbors\tParameters(n_neighbors=3)"
    return model,message


def get_decision_tree_classifier () :
    """Here you can optimize the parameters of the decision tree classifier"""

    model = DecisionTreeClassifier()
    message = "Decision Tree \tParameters(default)"
    return model,message


def get_logistic_regression_classifier_1 () :

    model = LogisticRegression(solver='lbfgs', max_iter=4000)
    message = "Logistic Regression\tParameters(solver='lbfgs', max_iter=4000)"
    return model, message


def get_logistic_regression_classifier_2 () :

    model = LogisticRegression(solver='lbfgs', max_iter=4000, class_weight={0: 1, 1: 5})
    message = "Logistic Regression\tsolver='lbfgs', max_iter=4000, class_weight={0: 1, 1: 5})"
    return model, message


def get_dummy_classifier () :

    model = DummyClassifier(strategy="stratified")
    message = "Dummy Classifier\tParameters((strategy='stratified')"
    return model, message


