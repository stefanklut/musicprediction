from music_feature_dict import music_feature_dict
from read_data import response_data
from benchmark import benchmark
from cross_val_split import create_split

def main(raw_response_data, raw_feature_data, n_folds = 10):
    header, feat_dict = music_feature_dict(raw_feature_data)
    responses = response_data(raw_response_data, feat_dict.keys())
    print(responses)

    for class_type in ['recognition', 'verification']:
        benchmark_values = benchmark(responses, class_type)
        song_id_split = create_split(responses, class_type, n_folds)
        # folds = create_folds(song_id_split, feat_dict, responses, class_type)
        # mean_eval, std_eval = cross_val(folds)
        # mean_eval_pca, std_eval_std = cross_val(folds, pca = True)

        # Write to csv

if __name__ == '__main__':
    main('tweedejaarsproject.csv', 'music_features.txt', n_folds = 100)
    print('scoop')
