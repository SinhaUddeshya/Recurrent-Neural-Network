import os
def get_subdirectories(path):
    return [name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))]
