'''
benchmark.py

find the benchmark (highest possible accuracy, precision, recall, F1-score
and specificity) for this classification task.

Input:
    data1:
        Data object created by read_data.py
    data_set:
        'recognition' or 'verification'
    threshold:
        value that indicates whether True or False is chosen, when the
        percentage of True is > treshold (default = 0.5)

Returns:
    a 4x5 arrray containing, in order, the accuracy, precision, recall,
    F1-score and specificity, for (top to bottom):
        - always choosing True
        - always choosing False
        - choosing True or False at random
        - choosing based on the percentage p of True per song, choosing True if
        p > threshold

'''
from read_data import *
from sklearn.metrics import confusion_matrix
from classifiers import measures
import time

def benchmark(p_data, data_set, threshold=0.5):

    start = time.time()

    responses = data1.get(data_set, 'is_response_correct')
    ids = data1.get(data_set, 'sound_cloud_id')

    combined_array = np.dstack((ids, responses))[0]

    combined_array = combined_array[combined_array[:,0].argsort()]

    combined_array = \
        np.array([combo for combo in combined_array])

    ids = combined_array[:, 0]
    responses = combined_array[:, 1]

    unique_ids, id_counts = np.unique(ids, return_counts=True)
    id_dict = dict(zip(unique_ids, id_counts))

    for key in id_dict:
        id_dict[key] = np.sum(combined_array[np.where(combined_array[:,0]\
            == key), 1]) / id_dict[key]

    # Create different matrices to test evaluation scores with
    true_vector = np.ones(len(responses))
    false_vector = np.zeros(len(responses))
    rand_vector = np.random.randint(2, size = len(responses))
    th_vector = \
        np.greater(np.array([id_dict[key] for key in ids]), threshold)

    # Create one big array to return
    true_results = \
        np.array([*measures(*confusion_matrix(responses, true_vector).ravel())])
    false_results = \
        np.array([*measures(*confusion_matrix(responses, false_vector).ravel())])
    random_results = \
        np.array([*measures(*confusion_matrix(responses, rand_vector).ravel())])
    threshold_results = \
        np.array([*measures(*confusion_matrix(responses, th_vector).ravel())])

    results = np.array([true_results, false_results, random_results, threshold_results])

    return results
