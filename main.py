'''
main.py

implements the main function, which combines other functions into a working
product.

Input:
    raw_response_data:
        The name of the csv file containing the response data
    raw_feature_data:
        The name of the csv file containing the music feature data
    n_folds:
        the number of folds the data will be divided in when using cross
        validation

Creates:
    a csv files recognition_results.csv and verification_results.csv,
    containing the measures and feature importance for different classifiers.

'''
import csv
import numpy as np
from music_feature_dict import music_feature_dict
from read_data import response_data
from benchmark import benchmark
from cross_val_split import create_split
from feat_resp_comb import create_folds
from classifiers import cross_val

def main(raw_response_data, raw_feature_data, n_folds = 10):
    header, feat_dict = music_feature_dict(raw_feature_data)
    responses = response_data(raw_response_data, feat_dict.keys())

    #for class_type in ['recognition', 'verification']:
    for class_type in ['recognition']:
        benchmark_values = benchmark(responses, class_type)
        song_id_split = create_split(responses, class_type, n_folds)
        folds = create_folds(song_id_split, feat_dict, responses, class_type)
        print('starting cross validation')
        mean_eval, std_eval = cross_val(folds)
        mean_measures = mean_eval[0][:5]
        std_measures = std_eval[0][:5]
        mean_features = mean_eval[0][5:]
        std_features = std_eval[0][5:]
        features_values =list(zip(header, mean_features, std_features))
        features_values.sort(key=lambda x: x[1], reverse = True)
        print(features_values[:10])
        # mean_eval_pca, std_eval_std = cross_val(folds, pca = True)

        f = open(class_type + '_results.csv', 'w')
        writer = csv.writer(f, delimiter = ';', lineterminator='\n')

        # Write benchmark values to csv
        f.write('benchmarks;\n')
        f.write(';Accuracy;Precision;Recall;F1-score;Specificity;\n')
        for i, t in enumerate(['All True;', 'All False;', 'Random True/False;',\
            'Threshold;']):
            f.write(t)
            writer.writerow(np.round(benchmark_values[i], decimals = 5))

if __name__ == '__main__':
    main('tweedejaarsproject.csv', 'music_features.txt', n_folds = 10)
