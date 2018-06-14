from read_data import data
from txt_to_dict import txt_to_dict
from file_connector import file_connect

feature_header, feature_dict = txt_to_dict('features.txt')
participant_data = data('first_pass.csv')

feature_matrix, responses = file_connect('verification',
                                         feature_dict, participant_data)
