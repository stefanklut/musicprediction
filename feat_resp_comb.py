'''
Filename: file_connector.py

A function that takes the song id split and returns matrices with the
responses and  music features combined.

'''
import numpy as np

def create_folds(song_id_split, feat_dict, responses, class_type):
	'''
	Takes the song_ids in the song id split and connects the music
	features with the response data to create the folds that are used for
	training and testing.

	Input:
		song_id_split:
			A list containing arrays of song ids
		feat_dict:
			A dictionary where the keys are the song ids and the values
			are the features of the music in a vector
		responses:
			A class containing the responses of the participants
		class_type:
			'recognition' or 'verification'

	'''
	response_tf = responses.get(class_type, 'is_response_correct')
	song_ids = responses.get(class_type, 'sound_cloud_id')
	unique_song_ids = np.unique(song_ids)
	# Dictionary comprehension that maps each id to its index in the list of ids
	id_location_dict = {sid:np.where(song_ids == sid)[0] for sid in \
		unique_song_ids}
	# Create a list that contains a list for every bucket of song ids in the
	# song_id_split list
	bucket_list = [[] for _ in range(len(song_id_split))]

	for i, fold in enumerate(song_id_split):
		for song_id in fold:
			# Acquire the list of answers for this song id
			r = response_tf[id_location_dict[song_id]]
			# Append the features of the song id to every response for that song
			# id, and append the result to the current bucket.
			bucket_list[i].append(np.hstack((np.tile(feat_dict[song_id], \
				(len(r), 1)), np.vstack(r))))
		# The bucket list is now an array with dimensions (sum of song id
		# features for all song ids in bucket)x
		# (length of the list of features + 1)
		bucket_list[i] = np.asarray([l for sid in bucket_list[i] for l in sid])
		# Shuffle it like a polaroid picture?!
		np.random.shuffle(bucket_list[i])
	return bucket_list

# For test purposes
if __name__ == '__main__':
	from music_feature_dict import music_feature_dict
	from read_data import response_data
	from cross_val_split import create_split
	import time
	start = time.time()
	h,m = music_feature_dict('music_features.txt')
	response_data = response_data('tweedejaarsproject.csv', m.keys())
	song_id_split = create_split(response_data, 'recognition', 10)
	f = create_folds(song_id_split, m, response_data, 'recognition')
	print(len(f[0]))
	print("Time:",time.time() - start)
