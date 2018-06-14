'''
filter_music.py

Reads in names of erroneous files from a file named bad_files.txt append
moves these files from the 'music' folder to an 'erroneous_music' folder.

'''
from os import listdir
import shutil

# Create list of filenames.
file_list = []
with open('bad_files.txt', 'r') as f:
    for line in f:
        line = line.strip('\n')
        file_list.append(line)
f.close()

# Move files to different folder.
os.mkdir('erroneous_music)
for filename in listdir('music'):
    if filename in file_list:
        shutil.move('music/'+filename, 'erroneous_music')
