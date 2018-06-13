import numpy as np


def file_connect(data_set, feature_data, participant_data_obj):
    """ Creates feature data matrix and corresponding participant answers."""

    responses = participant_data_obj.get(data_set, 'is_response_correct')
    song_id = participant_data_obj.get(data_set, 'sound_cloud_id')

    feature_length = len(list(feature_data.values())[0])
    feature_data_union = list(set(song_id) & set(feature_data.keys()))

    feature_matrix = []
    valid_song_id = []

    for i, key in enumerate(song_id):
        if key in feature_data_union:
            feature_matrix.append(feature_data[key])
            valid_song_id.append(i)

    return (np.asarray(feature_matrix), responses[valid_song_id])
