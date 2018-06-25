'''
Filename: read_data.py

Reads the data from a csv file into a dataframe for further uses.
'''
import numpy as np

class response_data:
    ''' Implements data class that saves participant data. '''
    def __init__(self, filename, good_file_ids, dl=';'):
        '''
        Inits the response_data class.

        Input:
            filename:
                The name of the csv file as a string containing the response data
            good_file_ids:
                List/iterator containing the ids of the files that will be used
            dl (default ';'):
                Delimiter for reading the csv file

        '''
        # Read in the csv file
        self.data_file = np.genfromtxt(filename, delimiter=dl, dtype=str)
        self.header = self.data_file[0]
        self.data_file = self.data_file[1:]
        # Remove files of which features cannot be extracted
        self.data_file = np.array([line for line in self.data_file \
                                    if int(line[-1]) in good_file_ids])
        self.__split_rec_ver()
        self.types = {'participant': 'int', 'song': 'int',
                      'start_point': 'str', 'recognition_time': 'float',
                      'is_response_correct': 'bool', 'verification_time': 'str',
                      'is_return_correct': 'str', 'timestamp': 'str',
                      'playlist': 'str', 'segment': 'int',
                      'sound_cloud_id': 'int'}

    def __split_rec_ver(self):
        ''' Function splits full dataset in recognition and verification sets. '''
        # Ko wat doe je hier allemaal ?????????????
        self.verification = self.data_file[self.data_file[:, 4] != 'NA']
        self.verification[self.verification[:, 4] == 'TRUE', 4] = 1
        self.verification[self.verification[:, 4] == 'FALSE', 4] = 0
        rec_false = self.data_file[self.data_file[:, 4] == 'NA']
        rec_false[:, 4] = 0
        rec_true = np.copy(self.verification)
        rec_true[:, 4] = 1
        self.recognition = np.concatenate((rec_false, rec_true))

    def get(self, class_type, var_name):
        '''
        Function to retreive specific data columns from specific dataset.

        Input:
            class_type:
                'recognition' or 'verification'
            var_name:
                The name of the column you want to return
        Output:
            returns the column that was specified with the var_name for the
            classification type that was specified with the class_type variable

        '''
        if var_name in self.header:
            # Get index from header
            [[var_index]] = np.where(self.header == var_name)
            var_type = self.types[var_name]
            if class_type == 'recognition':
                return self.recognition[:, var_index].astype(var_type)
            elif class_type == 'verification':
                return self.verification[:, var_index].astype(var_type)
        raise NameError('Invalid Name')

    def __str__(self):
        ''' When printing the class it is represented as the full data file. '''
        return(str(self.data_file))

if __name__ == '__main__':
    from music_feature_dict import *

    h,m = music_feature_dict('music_features.txt')
    d = response_data('tweedejaarsproject.csv', m.keys())
