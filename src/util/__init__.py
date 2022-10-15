import os


# Clears the files and directories in the given directory.
def clear_directory(directory_path):
    for d in os.listdir(directory_path):
        f_path = os.path.join(directory_path, d)
        if os.path.isfile(f_path):
            os.remove(f_path)
        elif os.path.isdir(f_path):
            clear_directory(f_path)
            os.rmdir(f_path)
