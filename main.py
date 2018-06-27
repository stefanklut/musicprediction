'''
Filename: main.py

DESCRIPTION

Project commisioned by:
	- John Ashley Burgoyne (ILLC)

By:
	- Dries Fransen (11041250)
	- Geerten Rijsdijk (11296720)
	- Hannah Min (11011580)
	- Ko Schoemaker (11347503)
	- Stefan Klut (11331720)

Part of the course "tweedejaarsproject" at the Uva, Bachelor of Science
Kunstmatige Intelligentie

'''

import csv
import sys
import numpy as np
from music_feature_dict import music_feature_dict
from read_data import response_data
from benchmark import benchmark
from cross_val_split import create_split
from feat_resp_comb import create_folds
from classifiers import cross_val

CLASSIFIERS = ['RandomForestClassifier', 'DecisionTreeClassifier', \
			 'AdaBoostClassifier', 'ExtraTreesClassifier']

def main(raw_response_data, raw_feature_data, n_folds = 10):
	'''
	Implements the main function, which combines other functions into a working
	product.

	Input:
		raw_response_data:
			The name of the csv file containing the response data
		raw_feature_data:
			The name of the txt file containing the music feature data
		n_folds (default 10):
			The number of folds the data will be divided in when using cross
			validation

	Output:
		Csv files recognition_results.csv and verification_results.csv,
		containing the measures and feature importance for different classifiers

	'''
	header, feat_dict = music_feature_dict(raw_feature_data)
	responses = response_data(raw_response_data, feat_dict.keys())

	# for class_type in ['recognition', 'verification']:
	for class_type in ['recognition', 'verification']:
		print('starting the', class_type, 'task.')
		# Open csv file to write to
		f = open(class_type + '_results.csv', 'w')
		writer = csv.writer(f, delimiter = ';', lineterminator='\n')

		# Compute benchmark values and write them to CSV.
		benchmark_values = benchmark(responses, class_type)
		f.write('benchmarks;\n')
		f.write(';Accuracy;Precision;Recall;F1-score;Specificity;\n')
		for i, t in enumerate(['All True;', 'All False;', 'Random True/False;',\
			'Baseline;']):
			f.write(t)
			writer.writerow(np.round(benchmark_values[i], decimals = 5))
		f.write('\n')
		writer.writerow(['Cross validation folds', str(n_folds)])
		f.write('\n')

		# Create cross validation folds
		song_id_split = create_split(responses, class_type, n_folds)
		folds = create_folds(song_id_split, feat_dict, responses, class_type)
		print('starting cross validation')
		mean_eval, std_eval = cross_val(folds)
		# mean_eval_pca, std_eval_pca = cross_val(folds, pca = True)

		# Write non-PCA results to csv
		f.write('classifiers without PCA;Measures;;;;;feature importance\n')
		f.write(';accuracy;precision;recall;F1-score;\
			specificity;1;2;3;4;5;6;7;8;9;10\n')
		for i in range(len(mean_eval)):

			f.write(CLASSIFIERS[i] + ';')
			mean_measures = np.round(mean_eval[i][:5], decimals = 5)
			std_measures = np.round(std_eval[i][:5], decimals = 5)
			mean_features = np.round(mean_eval[i][5:], decimals = 6)
			std_features = np.round(std_eval[i][5:], decimals = 6)
			measure_values = list(zip(mean_measures, std_measures))
			feature_values = list(zip(header, mean_features, std_features))
			feature_values.sort(key=lambda x: x[1], reverse = True)
			for i in range(5):
				f.write(str(measure_values[i][0]) + ' (' \
					+ str(measure_values[i][1]) + ');')
			for i in range(10):
				f.write(str(feature_values[i][0]) + ': ' + \
					str(feature_values[i][1]) + ' (' + \
						str(feature_values[i][2]) + ');')
			f.write('\n')

		# # Write PCA results to csv
		# f.write('classifiers with PCA;Measures;;;;;feature importance\n')
		# f.write(';accuracy;precision;recall;F1-score;\
		#	specificity;1;2;3;4;5;6;7;8;9;10\n')
		# for i in range(len(mean_eval_pca)):
		#
		# 	f.write(CLASSIFIERS[i] + ';')
		#
		# 	mean_measures = np.round(mean_eval_pca[i][:5], decimals = 5)
		# 	std_measures = np.round(std_eval_pca[i][:5], decimals = 5)
		# 	mean_features = np.round(mean_eval_pca[i][5:], decimals = 6)
		# 	std_features = np.round(std_eval_pca[i][5:], decimals = 6)
		# 	measure_values = list(zip(mean_measures, std_measures))
		# 	feature_values = list(zip(header, mean_features, std_features))
		# 	feature_values.sort(key=lambda x: x[1], reverse = True)
		#
		# 	for i in range(5):
		# 		f.write(str(measure_values[i][0]) + ' (' + \
		#	str(measure_values[i][1]) + ');')
		# 	for i in range(10):
		# 		f.write(str(feature_values[i][0]) + ': ' +\
		#	str(feature_values[i][1]) + ' (' + str(feature_values[i][2]) + ');')
		# 	f.write('\n')
		#
		# f.close()

if __name__ == '__main__':
	if len(sys.argv) != 4:
		print('Usage: main.py <participant_data> <music_features> <n_folds>')
	else:
		main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
