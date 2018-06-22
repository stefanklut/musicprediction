'''
read_data.py

Implements data class that saves participant data.
'''
import numpy as np


class data:
    """Can convert the CSV file containing responsedata from participants to
    usable python data. The class contains a full data array and two datasets
    with the recognition task data and verification task data. Class contains
    function to extract single columns from either recognition or verification
    data. Takes filename as argument."""

    def __init__(self, filename, good_file_ids, dl=';'):
        self.data_file = np.genfromtxt(filename, delimiter=dl, dtype=str)
        self.header = self.data_file[0]
        self.data_file = self.data_file[1:]
        print(len(self.data_file))
        # Remove files of which features cannot be extracted
        self.data_file = np.array([line for line in self.data_file \
                                    if int(line[-1]) in good_file_ids])
        print(len(self.data_file))
        self.__split_rec_ver()
        self.types = {'participant': 'int', 'song': 'int',
                      'start_point': 'str', 'recognition_time': 'float',
                      'is_response_correct': 'bool', 'verification_time': 'str',
                      'is_return_correct': 'str', 'timestamp': 'str',
                      'playlist': 'str', 'segment': 'int',
                      'sound_cloud_id': 'int'}

    def __split_rec_ver(self):
        # Function splits full dataset in recognition and verification sets.
        self.verification = self.data_file[self.data_file[:, 4] != 'NA']
        self.verification[self.verification[:, 4] == 'TRUE', 4] = 1
        self.verification[self.verification[:, 4] == 'FALSE', 4] = 0
        rec_false = self.data_file[self.data_file[:, 4] == 'NA']
        rec_false[:, 4] = 0
        rec_true = np.copy(self.verification)
        rec_true[:, 4] = 1
        self.recognition = np.concatenate((rec_false, rec_true))

    def get(self, data_set, var_name):
        # Function to retreive specific data columns from specific dataset.
        if var_name in self.header:
            [[var_index]] = np.where(self.header == var_name)
            var_type = self.types[var_name]
            if data_set == 'recognition':
                return self.recognition[:, var_index].astype(var_type)
            elif data_set == 'verification':
                return self.verification[:, var_index].astype(var_type)
        raise NameError('Invalid Name')

    def __str__(self):
        return(str(self.data_file))

if __name__ == '__main__':
    from music_feature_dict import *

    h,m = music_feature_dict('music_features.txt')
    d = data('tweedejaarsproject.csv', m.keys())
