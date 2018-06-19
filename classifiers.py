'''
classifiers.py

A great discription of the classifiers

'''
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


def classify_features(func, feature_matrix, responses, test_size=0.30, pca=False):
    features_train, features_test, responses_train, responses_test = \
        train_test_split(feature_matrix, responses, test_size=test_size)

    if pca:
        features_train, features_test = apply_pca(features_train, features_test)

    classify_function = func()
    classify_function.fit(features_train, responses_train)
    classify_prediction = classify_function.predict(features_test)
    feature_importance = enumerate(classify_function.feature_importances_)
    if pca:
        feature_importance = []
    return(*measures(*confusion_matrix(responses_test, classify_prediction).ravel()),
           sorted(feature_importance, key=lambda x:x[1])[-20:])

def measures(tn, fp, fn, tp):
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = (tp) / (fp + tp)
    recall = (tp) / (fn + tp)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    specificity = (tn) / (tn + fp)

    return (accuracy, precision, recall, f1_score, specificity)

def apply_pca(feat_train, feat_test):
    scaler = StandardScaler()
    scaler.fit(feat_train)

    feat_train = scaler.transform(feat_train)
    feat_test = scaler.transform(feat_test)

    pca = PCA(.95)
    pca.fit(feat_train)

    pca_train = pca.transform(feat_train)
    pca_test = pca.transform(feat_test)

    return pca_train, pca_test

"""
svc = SVC()
svc.fit(features, responses)

nusvc = NuSVC()
nusvc.fit(features, responses)

linearsvc = LinearSVC()
linearsvc.fit(features, responses)

ada_boost = AdaBoostClassifier()
ada_boost.fit(features, responses)

gradient_boost = GradientBoostingClassifier()
gradient_boost.fit(features, responses)

extra_tree = ExtraTreesClassifier()
extra_tree.fit(features, responses)

random_forest = RandomForestClassifier()
random_forest.fit(features, responses)

mlp = MLPClassifier()
mlp.fit(features, responses)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(features, responses)
"""
