import os 

def file_path_creation(directory_path): 
    paths = []
    for name in os.listdir(directory_path):
        paths.append(directory_path + '/' + name)
   
    return paths 

