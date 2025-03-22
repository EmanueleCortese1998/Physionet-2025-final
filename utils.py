import os 
from tqdm import tqdm
import wfdb 
import pandas as pd 
from datetime import datetime
import pickle 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
import csv 
def file_path_creation(directory_path): 
    paths = []
    for name in os.listdir(directory_path):
        paths.append(directory_path + '/' + name)
   
    return paths 
def data_creation(code15_data_path):
    """
    This function creates a database for the code15 data by reading .hea files from the specified directory.

    Parameters:
    code15_data_path (str): The path to the directory containing exam parts.

    Returns:
    tuple: A tuple containing the tensor of signals and a DataFrame of metadata.
    """
    code_15_df_rows = []   # Lista per raccogliere i dati del DataFrame

    # Check if the path exists
    if not os.path.exists(code15_data_path):
        raise FileNotFoundError(f"The specified path does not exist: {code15_data_path}")

    exam_parts = os.listdir(code15_data_path)

    # Use tqdm to show progress for exam parts
    for exam_part in tqdm(exam_parts, desc="Processing exam parts"):
        exam_part_record_path = os.path.join(code15_data_path, exam_part)

        # Check if the path is a directory
        if not os.path.isdir(exam_part_record_path):
            continue

        files_of_exam_part = [file for file in os.listdir(exam_part_record_path) if file.endswith('.hea')]

        # Use tqdm to show progress for files
        for file_of_exam_part in tqdm(files_of_exam_part, desc=f"Processing files in {exam_part}", leave=False):
            record_path = os.path.join(exam_part_record_path, file_of_exam_part.replace(".hea", ""))
            
            try:
                record = wfdb.rdrecord(record_path)
                tensor = torch.tensor(record.p_signal, dtype=torch.float32)

                # Aggiungi padding se necessario
                if tensor.size(0) < 4096:
                    padding = 4096 - tensor.size(0)
                    tensor = F.pad(tensor, (0, 0, 0, padding), mode='constant', value=0)

                # Aggiungi il tensore alla lista per concatenazione successiva
                yield tensor  # Usa yield per restituire i tensori uno alla volta

                # Raccogli i dati per il DataFrame
                new_row = {
                    'Age': int(record.comments[0].replace("Age: ", "")),
                    'Sex': record.comments[1].replace('Sex: ', ""),
                    'Label': record.comments[2].replace("Chagas label: ", "")
                }
                code_15_df_rows.append(new_row)

            except Exception as e:
                print(f"Error reading record {record_path}: {e}")

    # Concatenare tutti i tensori in un'unica operazione
    code_15_ECG = torch.cat(list(data_creation(code15_data_path)), dim=0)

    # Creare il DataFrame finale
    code_15_df = pd.DataFrame(code_15_df_rows)

    return code_15_ECG, code_15_df

def pad_tensor(tensor, target_size=1568):
    """Pads the tensor to the target size."""
    if tensor.size(1) < target_size:
        padding = target_size - tensor.size(1)
        tensor = F.pad(tensor, (0, 0, 0, padding), mode='constant', value=0)
    return tensor

def cut_tensor(tensor, tensor_max_dimension = 1568):

    if tensor.size(1) > tensor_max_dimension: 

        tensor = tensor[:,:tensor_max_dimension,:]
    return tensor 


def data_creation2(code15_data_path):
    """
    This function creates a database for the code15 data by reading .hea files from the specified directory.

    Parameters:
    code15_data_path (str): The path to the directory containing exam parts.

    Returns:
    tuple: A tuple containing the tensor of signals and a DataFrame of metadata.
    """
    code_15_df_rows = []  # List to collect DataFrame data
    all_tensors = []      # List to accumulate tensors
    all_tensors_not_padded = []
    # Check if the path exists
    if not os.path.exists(code15_data_path):
        raise FileNotFoundError(f"The specified path does not exist: {code15_data_path}")

    exam_parts = os.listdir(code15_data_path)

    # Use tqdm to show progress for exam parts
    for exam_part in tqdm(exam_parts, desc="Processing exam parts"):
        exam_part_record_path = os.path.join(code15_data_path, exam_part)

        # Check if the path is a directory
        if not os.path.isdir(exam_part_record_path):
            continue

        files_of_exam_part = [file for file in os.listdir(exam_part_record_path) if file.endswith('.hea')]

        # Use tqdm to show progress for files
        for file_of_exam_part in tqdm(files_of_exam_part, desc=f"Processing files in {exam_part}", leave=False):
            record_path = os.path.join(exam_part_record_path, file_of_exam_part.replace(".hea", ""))

            try:
                record = wfdb.rdrecord(record_path)
                tensor = torch.tensor(record.p_signal, dtype=torch.float32)
                tensor = torch.unsqueeze(tensor,0)
                
                # cutting if necessary
                tensor = cut_tensor(tensor)
                tensor = pad_tensor(tensor)
                

                all_tensors.append(tensor)  # Accumulate tensors

                # Collect data for the DataFrame
                new_row = {
                    'Age': int(record.comments[0].replace("Age: ", "")),
                    'Sex': record.comments[1].replace('Sex: ', ""),
                    'Label': record.comments[2].replace("Chagas label: ", "")
                }
                code_15_df_rows.append(new_row)

            except Exception as e:
                print(f"Error reading record {record_path}: {e}")

    # Concatenate all tensors once
    code_15_ECG = torch.cat(all_tensors, dim=0) if all_tensors else torch.empty(0)

    # Create the final DataFrame
    code_15_df = pd.DataFrame(code_15_df_rows)

    return code_15_ECG, code_15_df


def db_preparation(db): 

    db['Sex'] = db['Sex'].replace({'Male':1,'Female':0})
    db['Label'] = db['Label'].replace({'True':1,'False':0})
 


    return db


def saving_data(pickel_file,pandas_file,dir_for_saving): 
    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d %H%M%S")
    comments = input("Insert the comments")
    saving_path = dir_for_saving + '/' + date_time_string + '/'
    os.makedirs(os.path.dirname(saving_path), exist_ok=True)

    pandas_file.to_csv(saving_path + 'db.csv')
    with open(saving_path + 'comments.txt','w') as f:  
        f.write(comments)
    with open(saving_path + 'ECG.pkl','wb') as f: 
        pickle.dump(pickel_file,f)
    print('Saving succesful')


def load_data(dir_for_loading): 
    if not os.path.exists(dir_for_loading): 
        raise FileNotFoundError(f"la directory {saving_path} non esiste.")
    df = pd.read_csv(dir_for_loading + '/db.csv')
    with open(dir_for_loading+'/comments.txt','r') as f: 
        comments = f.read()
    with open(dir_for_loading + '/ECG.pkl', 'rb') as f: 
        pickle_data = pickle.load(f)

    print(comments)
    return pickle_data,df

def save_training_set(training_set,labels,saving_path): 
    now=datetime.now()    
    date_time_string = now.strftime("%Y-%m-%d %H%M%S")
    comments = input("Insert the comments")
    saving_path = saving_path + '/' +date_time_string + '/'
    os.makedirs(os.path.dirname(saving_path), exist_ok=True)
    with open(saving_path + 'comments.txt','w') as f: 
        f.write(comments)
    with open(saving_path + 'training_set.pkl','wb') as f: 
        pickle.dump(training_set,f)
    np.save(saving_path + 'labels.npy',labels)
    print("Saved succesfully")

def load_training_set(loading_path):
    if not os.path.exists(loading_path):
        raise FileNotFoundError(f"la directory {loading_path} non esiste.")
    labels = np.load(loading_path + '/labels.npy')
    with open(loading_path + '/training_set.pkl','rb') as f: 
        training_set = pickle.load(f)
    with open(loading_path+'/comments.txt','r') as f: 
        comments = f.read()
    print(comments)



    return training_set, labels
''''
class TimeSeriesCNN(nn.Module):
    def __init__(self, num_classes):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, stride=1,padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=0)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 390, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, 1)  # Single output for binary classification

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, sequence_length)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Use sigmoid for binary classification
        return x

class TimeSeriesCNN(nn.Module):
    def __init__(self, num_classes, kernel_size=3):
        super(TimeSeriesCNN, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=self.kernel_size, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # Placeholder per le dimensioni dell'output
        self.fc1 = None
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Cambia la forma in (batch_size, channels, sequence_length)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Appiattire il tensore

        if self.fc1 is None:
            # Calcola dinamicamente la dimensione dell'input per il layer fc1
            input_size = x.size(1)
            self.fc1 = nn.Linear(input_size, 128)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Usa sigmoid per la classificazione binaria

        return x'
        '''''

class TimeSeriesCNN(nn.Module):
    def __init__(self, num_classes):
        super(TimeSeriesCNN, self).__init__()
        
        # Convoluzione con kernel grande
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=128, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Convoluzione con kernel più piccolo
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=64, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(64)

        # Terza convoluzione con kernel ancora più piccolo
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=32, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Imposta direttamente la dimensione di input per fc1
        self.fc1 = nn.Linear(128 * 15, 128)  # Sostituisci 15 con la dimensione appropriata
        self.fc2 = nn.Linear(128, num_classes)  # Output per la classificazione

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Cambia la forma in (batch_size, channels, sequence_length)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Appiattire il tensore
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Usa sigmoid per la classificazione binaria
        return x