# go into all subfolders and create a text file inside them
# with the same name as the subfolder itself.

import os
path = 'C:\\Users\\Shivank\\Desktop\\RnnSpeech\\speech_commands_v0.01'

def get_subdirectories(path):
    return [name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))]

list_of_dir = []
list_of_dir = get_subdirectories(path)

os.chdir(os.path.join('speech_commands_v0.01'))

for subdir in list_of_dir:
    os.chdir(os.path.join(path, subdir))

    i = 0
    for filename in os.listdir('.'):
        os.rename(filename, subdir+'_'+((str)(i))+'.wav')
        i = i + 1

    os.chdir("../"+subdir) # to change back to dataset directory
