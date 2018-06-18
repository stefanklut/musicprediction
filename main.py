from read_data import data
from txt_to_dict import txt_to_dict
from file_connector import file_connect
from classifiers import classify_features

from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

feature_header, feature_dict = txt_to_dict('features.txt')
participant_data = data('first_pass.csv')

feature_matrix, responses = file_connect('recognition',
                                         feature_dict, participant_data)

accuracy, precision, recall, f1_score, specificity, feature_importance = \
    classify_features(RandomForestClassifier, feature_matrix, responses)
