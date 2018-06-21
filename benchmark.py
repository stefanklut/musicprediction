'''
benchmark.py

find the benchmark (highest possible accuracy, precision, recall, F1-score
and specificity) for this classification task.

'''
from read_data import *
from sklearn.metrics import confusion_matrix
from classifiers import measures
import time

def benchmark(data_set, omit, threshold=0.5):

    # np.set_printoptions(threshold=np.nan)
    start = time.time()
    data1 = data('tweedejaarsproject.csv')
    print('read data:', time.time() - start)

    responses = data1.get(data_set, 'is_response_correct')
    ids = data1.get(data_set, 'sound_cloud_id')

    # combined_array = np.array(list(zip(ids, responses)))
    combined_array = np.dstack((ids, responses))[0]

    combined_array = combined_array[combined_array[:,0].argsort()]

    start = time.time()
    combined_array = \
        np.array([combo for combo in combined_array if combo[0] not in omit])
    print('remove faulty files:', time.time() - start)

    ids = combined_array[:, 0]
    responses = combined_array[:, 1]

    unique_ids, id_counts = np.unique(ids, return_counts=True)
    id_dict = dict(zip(unique_ids, id_counts))

    for key in id_dict:
        id_dict[key] = np.sum(combined_array[np.where(combined_array[:,0]\
            == key), 1]) / id_dict[key]

    start = time.time()
    # Create different matrices to test evaluation scores with
    threshold_vector = \
        np.greater(np.array([id_dict[key] for key in ids]), threshold)
    print('create threshold vector:', time.time() - start)

    true_vector = np.ones(len(responses))
    false_vector = np.zeros(len(responses))
    rand_vector = np.random.randint(2, size = len(responses))

    # Output evaluation scores
    print('TRUE: ', \
        *measures(*confusion_matrix(responses, true_vector).ravel()))
    print('FALSE: ', \
        *measures(*confusion_matrix(responses, false_vector).ravel()))
    print('RANDOM: ', \
        *measures(*confusion_matrix(responses, rand_vector).ravel()))
    print('THRESHOLD: ',\
        *measures(*confusion_matrix(responses, threshold_vector).ravel()))

bad_files = []
with open('bad_files.txt', 'r') as f:
    for line in f:
        line = int(line.strip('\n')[:-4])
        bad_files.append(line)

benchmark('recognition', bad_files, threshold = 0.5)
