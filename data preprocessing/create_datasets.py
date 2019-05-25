
import os
import shutil
import random

path = "C:\\Users\\Shivank\\Desktop\\RnnSpeech\\collection"
training_path = "C:\\Users\\Shivank\\Desktop\\RnnSpeech\\train"
# training set (95% of collection) = 61,485 audio files

# get all files into a list
files = [file for file in os.listdir(path)]
# print(len(files))

amount = 61485
for x in range(amount):
    print((str)(x)+' files moved')
    chosen = random.choice(files)
    shutil.move(os.path.join(path, chosen), training_path)
    files.remove(chosen)

print(len(files))
