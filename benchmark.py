'''
benchmark.py

find the benchmark (highest possible accuracy, precision, recall, F1-score
and specificity) for this classification task.

'''
from read_data import *
from sklearn.metrics import confusion_matrix
from classifiers import measures

data = data('first_pass.csv')

responses = data.get('recognition', 'is_response_correct')
ids = data.get('recognition', 'sound_cloud_id')
# combined_array = np.array(list(zip(ids, responses)))
combined_array = np.dstack((ids, responses))[0]

combined_array = combined_array[combined_array[:,0].argsort()]

print(combined_array[:500])

# Create different matrices to test evaluation scores with
true_matrix = np.ones(len(responses))
false_matrix = np.zeros(len(responses))
rand_matrix = np.random.randint(2, size=len(responses))

# Output evaluation scores
print('TRUE: ', *measures(*confusion_matrix(responses, true_matrix).ravel()))
print('FALSE: ', *measures(*confusion_matrix(responses, false_matrix).ravel()))
print('RANDOM: ', *measures(*confusion_matrix(responses, rand_matrix).ravel()))
