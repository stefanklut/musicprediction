import numpy as np

class data:
    def __init__(self, filename, dl=';'):
        self.data_file = np.genfromtxt(filename, delimiter=dl, dtype=str)
        self.header = self.data_file[0]
        self.header[0] = 'id'
        self.data_file = self.data_file[1:]
        self.__split_rec_ver()
        # self.id = self.data_file[1:,0].astype(int)
        # self.participant = self.data_file[1:,1].astype(int)
        # self.song = self.data_file[1:,2].astype(int)
        # self.start_point = self.data_file[1:,3]
        # self.recognition_time = self.data_file[1:,4].astype(float)
        # self.is_response_correct = self.data_file[1:,5]
        # self.verification_time = self.data_file[1:,6]
        # self.is_return_correct = self.data_file[1:,7]
        # self.timestamp = self.data_file[1:,8]
        # self.playlist = self.data_file[1:,9]
        # self.segment = self.data_file[1:,10]
        # self.sound_cloud_id = self.data_file[1:,11].astype(int)

    def __str__(self):
        return(str(self.data_file))

    def __split_rec_ver(self):
        self.verification = self.data_file[self.data_file[:,5] != 'NA']
        rec_false = self.data_file[self.data_file[:,5] == 'NA']
        rec_false[:,5] = 'FALSE'
        rec_true = np.copy(self.verification)
        rec_true[:,5] = 'TRUE'
        self.recognition = np.concatenate((rec_false, rec_true))
        self.recognition[self.recognition[:,0].argsort()]

    def get(self, data_set, var_name):
        if var_name in self.header:
            var_index = self.header.index(var_name)
            if data_set == 'recognition':
                return self.recognition[var_index]
            elif data_set == 'verification':
                return self.verification[var_index]
        else:
            raise NameError('Invalid Name')

data = data('sample_data.csv')
print(data.get('recognition', 'participant'))
