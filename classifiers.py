from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

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
