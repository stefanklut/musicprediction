'''
file_connector.py

A great discription of the file_connector

'''
import numpy as np

# def feat_resp_comb(classification_type, response_data, participant_data_obj, noise=False,
# 				 lower_lim=-0.1, upper_lim=0.1):
# 	""" Creates feature data matrix and corresponding participant answers."""

# 	responses = participant_data_obj.get(classification_type, 'is_response_correct')
# 	song_id = participant_data_obj.get(classification_type, 'sound_cloud_id')

# 	feature_length = len(list(response_data.values())[0])
# 	feature_data_union = list(set(song_id) & set(response_data.keys()))

# 	feature_matrix = []
# 	valid_song_id = []

# 	for i, key in enumerate(song_id):
# 		if key in feature_data_union:
# 			feature_matrix.append(response_data[key])
# 			valid_song_id.append(i)

# 	feature_matrix = np.asarray(feature_matrix)
# 	if noise:
# 		feature_matrix += np.random.uniform(lower_lim, upper_lim, feature_matrix.shape)

# 	return (feature_matrix, responses[valid_song_id])


def create_folds(song_id_split, feat_dict, responses, class_type):
    response_tf = responses.get(class_type, 'is_response_correct')
    song_ids = responses.get(class_type, 'sound_cloud_id')
    unique_song_ids = np.unique(song_ids)
    id_location_dict = {sid:np.where(song_ids == sid)[0] for sid in unique_song_ids}
    bucket_list = [list(np.empty((0,len(feat_dict[song_ids[0]])+1))) for _ in range(len(song_id_split))]
    
    for i, fold in enumerate(song_id_split):
        for song_id in fold:
            r = response_tf[id_location_dict[song_id]]
            bucket_list[i].append(np.hstack((np.tile(feat_dict[song_id], (len(r), 1)), np.vstack(r))))
        bucket_list[i] = [l for sid in bucket_list[i] for l in sid]
    return bucket_list