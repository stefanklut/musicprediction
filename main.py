'''
Filename: main.py

DESCRIPTION

Project commisioned by:
    - John Ashley Burgonye (ILLC)

By:
    - Dries Fransen (11041250)
    - Geerten Rijsdijk (11296720)
    - Hannah Min (11011580)
    - Ko Schoemaker (11347503)
    - Stefan Klut (11331720)

Part of the course "tweedejaarsproject" at the UvA, Bachlor of Science
Kunstmatige Intelligentie.

'''
import csv
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

    #for class_type in ['recognition', 'verification']:
    for class_type in ['recognition']:
        print('starting the', class_type, 'task.')
        # Open csv file to write to
        f = open(class_type + '_results.csv', 'w')
        writer = csv.writer(f, delimiter = ';', lineterminator='\n')

        # Compute benchmark values and write them to CSV.
        benchmark_values = benchmark(responses, class_type)
        f.write('benchmarks;\n')
        f.write(';Accuracy;Precision;Recall;F1-score;Specificity;\n')
        for i, t in enumerate(['All True;', 'All False;', 'Random True/False;',\
            'Threshold;']):
            f.write(t)
            writer.writerow(np.round(benchmark_values[i], decimals = 5))
        f.write('\n')
        writer.writerow(['Cross validation folds', str(n_folds)])
        f.write('\n')

        song_id_split = create_split(responses, class_type, n_folds)
        folds = create_folds(song_id_split, feat_dict, responses, class_type)
        print('starting cross validation')
        mean_eval, std_eval = cross_val(folds)

        for i in range(len(mean_eval)):

            f.write(CLASSIFIERS[i] + ';')

            mean_measures = mean_eval[i][:5]
            std_measures = std_eval[i][:5]
            mean_features = np.round(mean_eval[i][5:], decimals = 6)
            std_features = np.round(std_eval[i][5:], decimals = 6)
            features_values = list(zip(header, mean_features, std_features))
            features_values.sort(key=lambda x: x[1], reverse = True)
            print(features_values[:10])
            writer.writerow(list(zip(mean_measures, std_measures)))
            writer.writerow(features_values[:10])

        f.close()




        # mean_eval_pca, std_eval_std = cross_val(folds, pca = True)

if __name__ == '__main__':
    main('tweedejaarsproject.csv', 'music_features.txt', n_folds = 2)
