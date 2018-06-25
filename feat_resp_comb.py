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
    # Dries wat doe je hier ??????????????????
	id_location_dict = {sid:np.where(song_ids == sid)[0] for sid in unique_song_ids}
	bucket_list = [list(np.empty((0,len(feat_dict[song_ids[0]])+1))) for _ in range(len(song_id_split))]

	for i, fold in enumerate(song_id_split):
		for song_id in fold:
			r = response_tf[id_location_dict[song_id]]
			bucket_list[i].append(np.hstack((np.tile(feat_dict[song_id], (len(r), 1)), np.vstack(r))))
		bucket_list[i] = np.asarray([l for sid in bucket_list[i] for l in sid])
		np.random.shuffle(bucket_list[i])
	return bucket_list
