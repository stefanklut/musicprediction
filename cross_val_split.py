'''
Filename: cross_val_split.py

A function to create folds for cross validation, the folds cannot have any overlapping
song ids. The number of folds is variable.

'''
import numpy as np
from collections import Counter
from read_data import *
import random

def create_split(response_data, class_type, n_buckets):
    '''
    Creates (n_buckets) buckets of roughly equal size, dividing data into sets where
    two sets never share a song_id.

    Input:
        response_data:
            Data object created by read_data.py
        class_type:
            'recognition' or 'verification'
        n_buckets:
            Number of buckets the data needs to be divided over

    Output:
        A list of n_buckets arrays, each containing a number of song_ids

    '''
    # Get the unique ids and their count
    ids = response_data.get(class_type, 'sound_cloud_id')
    counter_dict = Counter(ids)

    bucket_values = np.zeros(n_buckets)
    bucket_ids = [np.array([]) for _ in range(n_buckets)]

    # Shuffle the keys to make sure the folds are different each time
    keys  = list(counter_dict.keys())
    random.shuffle(keys)

    # For all song ids add the song id to the current lowest fold
    for song_id in keys:
        count = counter_dict[song_id]

        lowest_index = np.argmin(bucket_values)
        bucket_values[lowest_index] += count
        bucket_ids[lowest_index] = \
            np.append(bucket_ids[lowest_index], np.array([song_id]))
    return(bucket_ids)

# For test purposes
if __name__ == '__main__':
    from music_feature_dict import music_feature_dict
    import time
    h,m = music_feature_dict('music_features.txt')
    response_data = response_data('tweedejaarsproject.csv', m.keys())
    start = time.time()
    create_split(response_data, 'recognition', 10)
    print(time.time() - start)
