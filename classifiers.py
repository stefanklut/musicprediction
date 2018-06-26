'''
Filename: classifiers.py

Implementation off the cross validation with the training of the classifiers,
the calculation of the measures and transforming the features using PCA all
including as functions.

'''
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, \
							 ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def cross_val(folds, pca=False):
	'''
	A function that takes a list of feature matrices with their responses and
	applies cross validation to get more accurate scoring measures.

	Input:
		folds:
			A list of arrays, where each array is one of the folds used
			for cross validation. The array is a feature matrix with the
			responses as the last column
		pca (default False):
			Boolean whether or not to use pca to reduce the number of features

	Output:
		Returns both the mean and std for the scoring measures and feature
		importance in a single array
	'''
    classifiers_list = [RandomForestClassifier] #, DecisionTreeClassifier, \
    #                     AdaBoostClassifier, ExtraTreesClassifier]
    #
    means = []
    stds = []

    for classifier in classifiers_list:
        results = []
        for index, group in enumerate(folds):
			# Use the folds to create a train and a test set
            test_set = group
            train_set = folds[0:index] + folds[index+1:len(folds)]
            train_set = np.vstack(train_set)

			# Get the scoring measures and feature importance for the classifier
            a, p, r, f1, s, importance = train(classifier, train_set, test_set, pca=pca)
            result = [a, p, r, f1, s] + list(importance)

            results.append(result)
        classifier_means = np.mean(results, axis=0)
        classifier_stds = np.std(results, axis=0)
        means.append(classifier_means)
        stds.append(classifier_stds)
    return np.array(means), np.array(stds)

def train(func, train_set, test_set, pca):
	'''
	A function that trains a classifier and returns the scoring measures and the
	feature importance.

	Input:
		func:
			The name of the function that will be used to classified
		train_set:
			A matrix that is the training set with the responses as the final
			column and the rest of the matrix as the features
		test_set:
			A matrix that is the test set with the responses as the final column
			and the rest of the matrix as the features
		pca:
			Boolean whether or not to use pca to reduce the number of features

	Output:
		A tuple with scoring measures and a vector with the feature importance

	'''
	# Split into features and responses
    responses_train = train_set[:, -1]
    features_train = np.delete(train_set, -1, axis=1)
    responses_test = test_set[:, -1]
    features_test = np.delete(test_set, -1, axis=1)

    # apply PCA
    if pca:
        features_train, features_test = apply_pca(features_train, features_test)

	# Train the function with the training set
    classify_function = func()
    classify_function.fit(features_train, responses_train)
    classify_prediction = classify_function.predict(features_test)
    feature_importance = classify_function.feature_importances_

    return(*measures(*confusion_matrix(responses_test, classify_prediction).ravel()),
           feature_importance)

def measures(tn, fp, fn, tp):
	'''
	A function that returns several scoring measures based on the true
	negatives, false positives, false negatives and true positives.

	Input:
		tn:
			Number of true negatives
		fp:
			Number of false positives
		fn:
			Number of false negatives
		tp:
			Number of true positives

	Output:
		Returns a tuple with the following scoring measures:
		accuracy, precision, recall, f1_score, and specificity.

	'''
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = (tp) / (fp + tp)
    recall = (tp) / (fn + tp)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    specificity = (tn) / (tn + fp)

    return (accuracy, precision, recall, f1_score, specificity)

def apply_pca(feat_train, feat_test, pca_percentage=.95):
	'''
	Applies PCA to the training and test set to transform and reduce the number
	of features.

	Input:
		feat_train:
			The feature matrix of the training set
		feat_test:
			The feature matrix of the test set
		pca_percentage (default .95):
			Percentage of information that needs to be retained after PCA

	Output:
		Return the transformed feature matrices of the training and test set

	'''
	# Standerdize the features
    scaler = StandardScaler()
    scaler.fit(feat_train)
    feat_train = scaler.transform(feat_train)
    feat_test = scaler.transform(feat_test)

	# Train the PCA using the training set
    pca = PCA(pca_percentage)
    pca.fit(feat_train)

	# Transform the training and test matrices
    pca_train = pca.transform(feat_train)
    pca_test = pca.transform(feat_test)

    return pca_train, pca_test
