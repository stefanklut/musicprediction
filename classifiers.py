'''
classifiers.py

A great discription of the classifiers

'''
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


def classify_features(func, train_set, test_set, amount_of_features, pca=False):

    responses_train = train_set[:, -1]
    features_train = np.delete(array, -1, axis=1)
    responses_test = test_set[:, -1]
    features_test = np.delete(array, -1, axis=1)

    classify_function = func()
    classify_function.fit(features_train, responses_train)
    classify_prediction = classify_function.predict(features_test)
    feature_importance = enumerate(classify_function.feature_importances_)

    return(*measures(*confusion_matrix(responses_test, classify_prediction).ravel()),
           feature_importance)

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

    components = len(pca_train[1])

    return pca_train, pca_test, components

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
