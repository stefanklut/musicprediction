'''
results.py

Puts results of quality measures in csv file.

'''
import csv
import numpy as np
from read_data import data
from txt_to_dict import txt_to_dict
from file_connector import file_connect
from classifiers import classify_features

from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, \
                             ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


# Classifiers and measures
classifiers_list = [RandomForestClassifier, DecisionTreeClassifier, \
                    GradientBoostingClassifier, AdaBoostClassifier, \
                    ExtraTreesClassifier]
measures_list = ['Accuracy', 'Precision', 'Recall', 'f1_score', 'Specificity']

def data_processing(matlab_features, human_data):
    feature_header, feature_dict = txt_to_dict(matlab_features)
    participant_data = data(human_data)

    recognition_features, recognition_responses = file_connect('recognition',
                                             feature_dict, participant_data)
    verification_features, verification_responses = file_connect('verification',
                                             feature_dict, participant_data)

    return feature_header, recognition_features, recognition_responses, \
            verification_features, verification_responses

def evaluation(feature_matrix, responses, classifiers_list, pca=False):
    measures = []
    for classifier in classifiers_list:
        accuracy, precision, recall, f1_score, specificity, feature_importance = \
        classify_features(classifier, feature_matrix, responses, pca=pca)
        measures.append([accuracy, precision, recall, f1_score, specificity])
    return np.array(measures)


feature_header, recognition_features, recognition_responses, \
verification_features, verification_responses = \
data_processing('features.txt', 'first_pass.csv')

print("RECOGNITION TASK")
print(measures_list)
print(evaluation(recognition_features, recognition_responses, classifiers_list))

print("RECOGNITION TASK PCA")
print(measures_list)
print(evaluation(recognition_features, recognition_responses, classifiers_list, pca=True))

print("VERIFICATION TASK")
print(measures_list)
print(evaluation(verification_features, verification_responses, classifiers_list))

print("VERIFICATION TASK PCA")
print(measures_list)
print(evaluation(verification_features, verification_responses, classifiers_list, pca=True))

#names_feature_importance = [(feature_header[i], value) for i, value in reversed(feature_importance)]
#print("Measures", accuracy, precision, recall, f1_score, specificity)
