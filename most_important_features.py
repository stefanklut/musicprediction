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
import numpy as np

feature_header, feature_dict = music_feature_dict('features.txt')
participant_data = response_data('tweedejaarsproject.csv')


feature_matrix, responses = file_connect('recognition', feature_dict, participant_data)

# accuracy, precision, recall, f1_score, specificity, feature_importance = \
# 	classify_features(RandomForestClassifier, feature_matrix, responses, 1)

# names_feature_importance = [(feature_header[i], value) for i, value in reversed(feature_importance)]
# important_features_set = [feature_header[i] for i, _ in reversed(feature_importance)]

def classification(connect_args1, connect_args2, connect_args3, classify_args1, classify_args2, classify_args3, classify_args4):
	# start = time.time()
	# feature_matrix, responses = file_connect(connect_args1, connect_args2, connect_args3)

	accuracy, precision, recall, f1_score, specificity, feature_importance = classify_features(classify_args1, classify_args2, classify_args3, classify_args4)

	important_features_set.append([feature_header[i] for i, _ in reversed(feature_importance)])
	# print(time.time()-start)


print('taking off...')

start = time.time()

important_features_set = []
n_loops = 1
n_threads = 2

for _ in range(n_loops):
	thread_list = []
	start1 = time.time()
	for i in range(n_threads):
		thread_list.append(Thread(target=classification, \
							args=('recognition', feature_dict, participant_data, RandomForestClassifier, feature_matrix, responses, 10)))
		# t2 = Thread(target=classification, \
		# 			args=('recognition', feature_dict, participant_data, RandomForestClassifier, feature_matrix, responses, 10))
		# t3 = Thread(target=classification, \
		# 			args=('recognition', feature_dict, participant_data, RandomForestClassifier, feature_matrix, responses, 10))
	for t in thread_list:
		t.start()
		# t2.start()
		# t3.start()
	for t in thread_list:
		t.join()
	# t2.join()
	# t3.join()
	print(n_threads, 'thread' + ('s' if n_threads>1 else '') + ' finished in', '%.3f'%(time.time()-start1), 'seconds')

t = time.time()-start
print('\ntotal time:', '%.3f'%t)
print('average time:', '%.3f'%(t/(n_loops*n_threads)), '\n')

start = time.time()
classification('recognition', feature_dict, participant_data, RandomForestClassifier, feature_matrix, responses, 10)
t1 = time.time()-start
print('no thread time:', '%.3f'%t1, '\n')
print('total approximate time save:', '%.3f'%(t1*n_loops*n_threads-t), 'seconds\n\n')

important_features_set = np.array([item for sublist in important_features_set for item in sublist])
feats, counts = np.unique(important_features_set, return_counts=True)
zipped_list = list(zip(feats, counts))
zipped_list.sort(key=lambda x: x[1], reverse = True)
print(zipped_list)


# print(len(important_features_set))
