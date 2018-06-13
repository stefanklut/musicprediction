import numpy as np


class data:
    """Can convert the CSV file containing responsedata from participants to
    usable python data. The class contains a full data array and two datasets
    with the recognition task data and verification task data. Class contains
    function to extract single columns from either recognition or verification
    data. Takes filename as argument."""

    def __init__(self, filename, dl=';'):
        self.data_file = np.genfromtxt(filename, delimiter=dl, dtype=str)
        self.header = self.data_file[0]
        self.header[0] = 'id'
        self.data_file = self.data_file[1:]
        self.__split_rec_ver()
        self.types = {'id': 'int', 'participant': 'int', 'song': 'int',
                      'start_point': 'str', 'recognition_time': 'float',
                      'is_response_correct': 'bool', 'verification_time': 'str',
                      'is_return_correct': 'str', 'timestamp': 'str',
                      'playlist': 'str', 'segment': 'int',
                      'sound_cloud_id': 'int'}

    def __split_rec_ver(self):
        # Function splits full dataset in recognition and verification sets.
        self.verification = self.data_file[self.data_file[:, 5] != 'NA']
        self.verification[self.verification[:, 5] == 'TRUE', 5] = 1
        self.verification[self.verification[:, 5] == 'FALSE', 5] = 0
        rec_false = self.data_file[self.data_file[:, 5] == 'NA']
        rec_false[:, 5] = 0
        rec_true = np.copy(self.verification)
        rec_true[:, 5] = 1
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
