import sys
sys.dont_write_bytecode = True

import os
import shutil

class DirectoryProcessor:
    
    @staticmethod
    def get_all_files(directory, include_sub_dir=False):
        if include_sub_dir:
            path_list = [os.path.join(root,file) for root, _, files in os.walk(directory, topdown=True) for file in files]
        else:
            path_list = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

        return path_list

    @staticmethod
    def get_only_files(directory, extension, include_sub_dir=False):
        if include_sub_dir:
            path_list = [os.path.join(root,file) for root, _, files in os.walk(directory, topdown=True) for file in files]
            path_list = [path for path in path_list if any(ext in path for ext in extension)]
        else:
            path_list = [os.path.join(directory, file) for file in os.listdir(directory) if any(file.endswith(ext) for ext in extension)]

        return path_list
    
    @staticmethod
    def move_file(source, destination):
        destination_dir = os.path.dirname(destination)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        
        shutil.move(source, destination)
        pass