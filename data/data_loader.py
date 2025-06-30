import os
import numpy as np
from sklearn.model_selection import train_test_split

UCI_HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATASET_DIR = "UCI_HAR_Dataset"

def check_and_prompt_download():
    if not os.path.exists(DATASET_DIR):
        print(f"[INFO] UCI HAR dataset not found in '{DATASET_DIR}'.")
        print("[ACTION] Please download the dataset using the following command:")
        print(f"curl -L {UCI_HAR_URL} -o UCI_HAR_Dataset.zip")
        print("unzip UCI_HAR_Dataset.zip -d UCI_HAR_Dataset")
        print("rm UCI_HAR_Dataset.zip  # Optional: remove zip after extraction")
        return False
    return True

def load_signals(subdir, signal_type):
    base_path = os.path.join(DATASET_DIR, 'UCI HAR Dataset', subdir, 'Inertial Signals')
    signal_files = [
        f"{signal_type}_x.txt", 
        f"{signal_type}_y.txt", 
        f"{signal_type}_z.txt"
    ]
    signals = []
    for file in signal_files:
        with open(os.path.join(base_path, file), 'r') as f:
            signals.append([np.array(row.split(), dtype=np.float32) for row in f])
    return np.transpose(np.array(signals), (1, 2, 0))

def load_labels(subdir):
    label_file = os.path.join(DATASET_DIR, 'UCI HAR Dataset', subdir, f'y_{subdir}.txt')
    with open(label_file, 'r') as f:
        return np.array([int(row.strip()) for row in f])

def load_subjects(subdir):
    subject_file = os.path.join(DATASET_DIR, 'UCI HAR Dataset', subdir, f'subject_{subdir}.txt')
    with open(subject_file, 'r') as f:
        return np.array([int(row.strip()) for row in f])

def normalize_data(X):
    mean = np.mean(X, axis=(0, 1), keepdims=True)
    std = np.std(X, axis=(0, 1), keepdims=True)
    return (X - mean) / (std + 1e-8)

def preprocess_data(X, y, subjects):
    X = normalize_data(X)
    subject_data = {}
    unique_subjects = np.unique(subjects)
    for subj in unique_subjects:
        subject_mask = (subjects == subj)
        subject_data[subj] = {
            'X': X[subject_mask],
            'y': y[subject_mask],
            'subject_id': subj
        }
    return subject_data

def load_and_preprocess():
    if not check_and_prompt_download():
        return None

    # Load and combine train/test data
    X_train = load_signals('train', 'body_acc')
    X_test = load_signals('test', 'body_acc')
    X = np.concatenate((X_train, X_test), axis=0)
    
    y_train = load_labels('train')
    y_test = load_labels('test')
    y = np.concatenate((y_train, y_test), axis=0)
    
    subjects_train = load_subjects('train')
    subjects_test = load_subjects('test')
    subjects = np.concatenate((subjects_train, subjects_test), axis=0)
    
    subject_data = preprocess_data(X, y, subjects)
    
    processed_data = {}
    for subj, data in subject_data.items():
        X_subj, X_val, y_subj, y_val = train_test_split(
            data['X'], data['y'], 
            test_size=0.2, 
            random_state=42
        )
        processed_data[subj] = {
            'train': {'X': X_subj, 'y': y_subj},
            'validate': {'X': X_val, 'y': y_val}
        }
    
    return processed_data

# Example usage
if __name__ == "__main__":
    subject_data = load_and_preprocess()
    if subject_data and isinstance(subject_data, dict):
        print(f"Loaded data for {len(subject_data)} subjects")
        for subj, data in list(subject_data.items())[:3]:  # Show first 3 subjects
            print(f"\nSubject {subj}:")
            print(f"  Training samples: {len(data['train']['y'])}")
            print(f"  Validation samples: {len(data['validate']['y'])}")
            print(f"  Activity distribution: {np.bincount(data['train']['y'])}")
    else:
        print("[ERROR] Dataset not loaded. Please download and extract the dataset as instructed above.")
