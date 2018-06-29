% Script to extract statistical features from wav-file to a csv-file
% Go to folder with all music files to be extracted and run the script

struct = mirfeatures('Folder', 'Stat')
mirexport('music_features.txt', struct)