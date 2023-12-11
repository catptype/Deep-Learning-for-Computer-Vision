import os
import shutil

class DirectoryProcessor:
    """
    Utility class for processing directories and files.

    Methods:
        get_all_files(directory, include_sub_dir=False): Retrieves a list of all files in a directory, including subdirectories.
        get_only_files(directory, extension, include_sub_dir=False): Retrieves a list of files with specified extensions in a directory, optionally including subdirectories.
        move_file(source, destination): Moves a file from the source path to the destination path.

    Example:
        ```python
        # Example usage of DirectoryProcessor class
        files = DirectoryProcessor.get_all_files(directory="/path/to/directory", include_sub_dir=True)
        image_files = DirectoryProcessor.get_only_files(directory="/path/to/images", extension=[".jpg", ".png"], include_sub_dir=False)
        DirectoryProcessor.move_file(source="/path/to/source/file.txt", destination="/path/to/destination/file.txt")
        ```

    """
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