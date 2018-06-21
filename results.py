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
from benchmark import benchmark

from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, \
                             ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def evaluation(groups, human_data):
    measures_list = ['Accuracy', 'Precision', 'Recall', 'f1_score', 'Specificity']

    data_obj = data(human_data)
    benchmark_recognition = benchmark(data_obj, 'recognition', threshold = 0.5)
    benchmark_verification = benchmark(data_obj, 'verification', threshold = 0.5)

    # Cross validation loop
    classifiers_list = [RandomForestClassifier, DecisionTreeClassifier, \
                        GradientBoostingClassifier, AdaBoostClassifier, \
                        ExtraTreesClassifier]
    for classifier in classifiers_list:
        measures = []
        importance = []
        for index, group in enumerate(groups):
            test_set = group
            train_set = groups[0:index] + groups[index+1:len(groups)]
            train_set = np.vstack(train_set)
            a, p, r, f1, s, importance = train(classifier, train_set, test_set)
            measures.append([a, p, r, f1, s])
            importance.append([importance])
        avg_measures = np.mean(measures, axis=1)
        avg_importance = np.mean(importance, axis=1)
        std_importance = np.std(importance, axis=1)
        ind = reversed(np.argsort(avg_importance))[:10]
                

    # with open('recognition_results.csv', 'w') as csvfile:
    #     rec_writer = csv.writer('recognition_results', delimiter=',')
    #     rec_writer.writerow(['Benchmark Recognition Task'])
    #
    # with open('recognition_results.csv', 'w') as csvfile:
    #     ver_writer = csv.writer('verification_results', delimiter=',')
    #     ver_writer.writerow(['Benchmark Verification Task'])
    return True

def train(classifier, train_set, test_set):

    accuracy, precision, recall, f1_score, specificity, feature_importance = \
    classify_features(classifier, train_set, test_set)
    return accuracy, precision, recall, f1_score, specificity, feature_importance
