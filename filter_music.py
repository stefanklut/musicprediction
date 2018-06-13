from os import listdir
import shutil

file_list = []
with open('bad_files.txt', 'r') as f:
    for line in f:
        line = line.strip('\n')
        file_list.append(line)
f.close()


for filename in listdir('music'):
    if filename in file_list:
        shutil.move('music/'+filename, 'erroneous_music')
