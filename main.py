'''
main.py

A great discription of the main: such main, many python.

'''
from read_data import data
from txt_to_dict import txt_to_dict
from file_connector import file_connect
from classifiers import classify_features

from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, \
							 ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import time
from multiprocessing import Process, Manager
from threading import Thread

feature_header, feature_dict = txt_to_dict('features.txt')
participant_data = data('first_pass.csv')


feature_matrix, responses = file_connect('recognition', feature_dict, participant_data)

# accuracy, precision, recall, f1_score, specificity, feature_importance = \
# 	classify_features(RandomForestClassifier, feature_matrix, responses, 1)

# names_feature_importance = [(feature_header[i], value) for i, value in reversed(feature_importance)]
# important_features_set = [feature_header[i] for i, _ in reversed(feature_importance)]

def paardenstront(connect_args1, connect_args2, connect_args3, classify_args1, classify_args2, classify_args3, classify_args4):
	# start = time.time()
	# feature_matrix, responses = file_connect(connect_args1, connect_args2, connect_args3)

	accuracy, precision, recall, f1_score, specificity, feature_importance = classify_features(classify_args1, classify_args2, classify_args3, classify_args4)

	important_features_set.append([feature_header[i] for i, _ in reversed(feature_importance)])
	# print(time.time()-start)


print('taking off...')

start = time.time()

important_features_set = []
aantal_paarden = 3
for _ in range(aantal_paarden):
	start1 = time.time()
	t1 = Thread(target=paardenstront, \
				args=('recognition', feature_dict, participant_data, RandomForestClassifier, feature_matrix, responses, 10))
	t2 = Thread(target=paardenstront, \
				args=('recognition', feature_dict, participant_data, RandomForestClassifier, feature_matrix, responses, 10))
	t3 = Thread(target=paardenstront, \
				args=('recognition', feature_dict, participant_data, RandomForestClassifier, feature_matrix, responses, 10))
	t1.start()
	t2.start()
	t3.start()
	t1.join()
	t2.join()
	t3.join()
	print('3 threads finished in', '%.3f'%(time.time()-start1), 'seconds')

t = time.time()-start
print('\ntotal time:', '%.3f'%t)
print('average time:', '%.3f'%(t/(aantal_paarden*3)), '\n')

start = time.time()
paardenstront('recognition', feature_dict, participant_data, RandomForestClassifier, feature_matrix, responses, 10)
t1 = time.time()-start
print('no thread time:', '%.3f'%t1, '\n')
print('total approximate time save:', (t1*aantal_paarden*3-t), 'seconds\n\n')

important_features_set = set([item for sublist in important_features_set for item in sublist])
print(important_features_set)
print(len(important_features_set))