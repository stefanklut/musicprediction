'''
file_connector.py

A great discription of the file_connector

'''
import numpy as np
from read_data import data
from txt_to_dict import txt_to_dict

def connect(song_ids, data_set, feature_data, participant_data_obj):
    responses = participant_data_obj.get(data_set, 'is_response_correct')
    song_id = participant_data_obj.get(data_set, 'sound_cloud_id')
    id_responses = zip(song_id, responses)

    return_list = []
    for song_id_array in array_list:
        feature_array = np.array([])
        for song_id in song_id_array:
            feature_array = 


feature_header, feature_dict = txt_to_dict('features.txt')
participant_data = data('tweedejaarsproject.csv')
print(connect(np.array([147526948, 147526955], [147526959, 147527019, 147527022]), 'recognition', feature_dict, participant_data))


