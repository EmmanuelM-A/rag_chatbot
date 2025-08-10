"""Helper functions and classes"""

import os

def does_file_exist(file_path):
    if os.path.exists(file_path):
        print(f"The file {file_path} exists!")
    else:
        print(f"The file {file_path} exists!")