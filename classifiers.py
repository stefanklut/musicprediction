from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def classify_features(func, feature_matrix, responses, test_size=0.30):
    feature_matrix_train, feature_matrix_test, responses_train, \
        responses_test = \
        train_test_split(feature_matrix, responses, test_size=test_size)

    classify_function = func()
    classify_function.fit(feature_matrix_train, responses_train)
    classify_prediction = classify_function.predict(feature_matrix_test)
    feature_importance = classify_function.feature_importances_
    tn, fp, fn, tp = confusion_matrix(responses_test, classify_prediction).ravel()

    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = (tp) / (fp + tp)
    recall = (tp) / (fn + tp)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    specificity = (tn) / (tn + fp)

    return (accuracy, precision, recall, f1_score, specificity, sorted(feature_importance)[20:])

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
