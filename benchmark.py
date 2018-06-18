'''
benchmark.py

find the benchmark (highest possible accuracy, precision, recall, F1-score
and specificity) for this classification task.

'''
from read_data import *
from sklearn.metrics import confusion_matrix
from classifiers import measures
import time

# np.set_printoptions(threshold=np.nan)
CATCHINESS_THRESHOLD = 0.5

data = data('first_pass.csv')

responses = data.get('recognition', 'is_response_correct')
ids = data.get('recognition', 'sound_cloud_id')
# combined_array = np.array(list(zip(ids, responses)))
combined_array = np.dstack((ids, responses))[0]

combined_array = combined_array[combined_array[:,0].argsort()]

unique_ids, id_counts = np.unique(ids, return_counts=True)
id_dict = dict(zip(unique_ids, id_counts))

start = time.time()
for key in id_dict:
    id_dict[key] = np.sum(combined_array[np.where(combined_array[:,0] == key), 1]) / id_dict[key]

print(time.time()-start)

threshold_vector = np.array([])
for id in ids:
    threshold_vector = np.append(threshold_vector, id_dict[id] > CATCHINESS_THRESHOLD)

print(np.sort(threshold_vector))


# Create different matrices to test evaluation scores with
true_vector = np.ones(len(responses))
false_vector = np.zeros(len(responses))
rand_vector = np.random.randint(2, size=len(responses))

# Output evaluation scores
print('TRUE: ', *measures(*confusion_matrix(responses, true_vector).ravel()))
print('FALSE: ', *measures(*confusion_matrix(responses, false_vector).ravel()))
print('RANDOM: ', *measures(*confusion_matrix(responses, rand_vector).ravel()))
print('THRESHOLD: ', *measures(*confusion_matrix(responses, threshold_vector).ravel()))
