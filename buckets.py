'''
buckets.py

creates (n_buckets) buckets of roughly equal size, dividing data into sets where
two sets never share a song_id.

'''
import numpy as np
from collections import Counter
from read_data import *
import random

def create_buckets(part_data, data_set, n_buckets):

    ids = part_data.get(data_set, 'sound_cloud_id')

    counter_dict = Counter(ids)

    bucket_values = np.zeros(n_buckets)
    bucket_ids = []
    for i in range(n_buckets):
        bucket_ids.append(np.array([]))

    keys  = list(counter_dict.keys())
    random.shuffle(keys)

    for song_id in keys:
        count = counter_dict[song_id]

        lowest_index = np.argmin(bucket_values)

        bucket_values[lowest_index] += count
        bucket_ids[lowest_index] = \
            np.append(bucket_ids[lowest_index], np.array([song_id]))

    print(bucket_values)

    return(bucket_ids)

# part_data = data('tweedejaarsproject.csv')
# create_buckets(part_data, 'recognition', 10)
