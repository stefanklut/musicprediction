'''
Filename: txt_to_dict.py

A function to read in the music feature for a txt file that was generated
by Matlab.

'''
import numpy as np
from sklearn.preprocessing import normalize

def music_feature_dict(filename):
    '''
    Reads in a txt file with the music feature and returns the names of
    these feature and the values of these features for each song.

    Input:
        filename:
            The filename of the music features file as a string

    Output:
        A dict containing a list of the names of the features, and a
        dictionary with song ids as keys and the value are the features
        of the music in a vector

    '''
    header = ''
    # Get the names of the features we will be extracting
    with open(filename) as f:
        header = next(f).split('\t')[1:]
    # Load the .txt file generated by matlab into python, omitting the header
    textfile_as_array = np.loadtxt(filename, skiprows=1, \
        converters={0:lambda x: x[:-4]})
    song_ids = textfile_as_array[:,0].astype(int)
    # Remove any feature column that contains a NaN; this removes 35 columns
    not_normalized = textfile_as_array[:,~np.any(np.isnan(textfile_as_array), \
        axis=0)]
    # Normalize
    features = normalize(not_normalized, norm='l1', axis=0)
    # Return the song ids and the features as a dictionary
    return(header, dict(zip(song_ids, features)))
