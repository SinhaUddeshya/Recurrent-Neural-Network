import shutil
import os
from get_subfolder import get_subdirectories

path = 'C:\\Users\\Shivank\\Desktop\\RnnSpeech\\speech_commands_v0.01'
destination_path = 'C:\\Users\\Shivank\\Desktop\\RnnSpeech\\collection'

list_of_dir = []
list_of_dir = get_subdirectories(path)
os.chdir(os.path.join('speech_commands_v0.01'))

for subdir in list_of_dir:
    # go inside the subfolder
    os.chdir(os.path.join(path, subdir))

    # move each audio file into the collection
    for wav_file in os.listdir('.'):
        shutil.move(wav_file, destination_path)

    # go back to dataset directory
    os.chdir("../"+subdir)
