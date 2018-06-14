import numpy as np
from sklearn.preprocessing import normalize
        
def txt_to_dict(filename):
    header = ''
#     Get the names of the features we will be extracting
    with open(filename) as f:
        header = next(f).split('\t')[1:]
#     Load the .txt file generated by matlab into python, omitting the header
    textfile_as_array = np.loadtxt(filename, skiprows=1, converters={0:lambda x: x[:-4]})
    song_ids = textfile_as_array[:,0].astype(int)
#     Remove any feature column that contains a NaN; this removes 35 columns
    not_normalized = textfile_as_array[:,~np.any(np.isnan(textfile_as_array), axis=0)]
#     Normalize
    features = normalize(not_normalized, norm='l1', axis=0)
#     Return the song ids and the features as a dictionary
    return(header, dict(zip(song_ids, features)))