import numpy as np

class data:
    def __init__(self, filename, dl=';'):
        self.data_file = np.genfromtxt(filename, delimiter=dl, dtype=str)
        self.header = self.data_file[0]
        self.header[0] = 'id'
        self.data_file = self.data_file[1:]
        self.__split_rec_ver()
        self.types = {'id': 'int', 'participant': 'int', 'song': 'int', \
            'start_point': 'str', 'recognition_time': 'float', \
            'is_response_correct': 'bool', 'verification_time': 'str', \
            'is_return_correct': 'str', 'timestamp': 'str', 'playlist': 'str', \
            'segment': 'int', 'sound_cloud_id': 'int'}

    def __str__(self):
        return(str(self.data_file))

    def __split_rec_ver(self):
        self.verification = self.data_file[self.data_file[:,5] != 'NA']
        self.verification[self.verification[:,5] == 'TRUE',5] = True
        self.verification[self.verification[:,5] == 'FALSE',5] = False
        rec_false = self.data_file[self.data_file[:,5] == 'NA']
        rec_false[:,5] = False
        rec_true = np.copy(self.verification)
        rec_true[:,5] = True
        self.recognition = np.concatenate((rec_false, rec_true))

    def get(self, data_set, var_name):
        if var_name in self.header:
            [[var_index]] = np.where(self.header == var_name)
            var_type = self.types[var_name]
            if data_set == 'recognition':
                return self.recognition[:,var_index].astype(var_type)
            elif data_set == 'verification':
                return self.verification[:,var_index].astype(var_type)
        raise NameError('Invalid Name')
