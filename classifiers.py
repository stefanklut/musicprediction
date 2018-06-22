'''
classifiers.py

A great discription of the classifiers

'''
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def cross_val(folds, pca=False):
    classifiers_list = [RandomForestClassifier, DecisionTreeClassifier, \
                        GradientBoostingClassifier, AdaBoostClassifier, \
                        ExtraTreesClassifier]

    for classifier in classifiers_list:
        mean_eval = []
        std_eval = []

        for index, group in enumerate(groups):
            test_set = group
            train_set = groups[0:index] + groups[index+1:len(groups)]
            train_set = np.vstack(train_set)

            a, p, r, f1, s, importance = train(classifier, train_set, test_set, pca=pca)
            measures.append([a, p, r, f1, s])
            importance.append([importance])

        means = np.hstack((np.mean(measures, axis=1),  np.mean(importance, axis=1)))
        stds = np.hstack((np.std(measures, axis=1), np.std(importance, axis=1)))
        mean_eval.append(means)
        std_eval.append(stds)
    return np.array(mean_eval), np.array(std_eval)

def train(func, train_set, test_set, pca):
    responses_train = train_set[:, -1]
    features_train = np.delete(train_set, -1, axis=1)
    responses_test = test_set[:, -1]
    features_test = np.delete(test_set, -1, axis=1)

    # apply PCA
    if pca:
        features_train, features_test = apply_pca(features_train, features_test)

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

def apply_pca(feat_train, feat_test, pca_percentage=.95):
    scaler = StandardScaler()
    scaler.fit(feat_train)

    feat_train = scaler.transform(feat_train)
    feat_test = scaler.transform(feat_test)

    pca = PCA(pca_percentage)
    pca.fit(feat_train)

    pca_train = pca.transform(feat_train)
    pca_test = pca.transform(feat_test)

    return pca_train, pca_test
