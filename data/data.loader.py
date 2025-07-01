import os
import numpy as np

DATASET_DIR = "UCI HAR Dataset"

def load_signals(subdir):
    # Load all 9 sensor signals: body_acc, body_gyro, total_acc (x, y, z)
    signal_types = ['body_acc', 'body_gyro', 'total_acc']
    axes = ['x', 'y', 'z']
    signals = []
    for stype in signal_types:
        for axis in axes:
            file_path = os.path.join(DATASET_DIR, subdir, 'Inertial Signals', f"{stype}_{axis}_{subdir}.txt")
            with open(file_path, 'r') as f:
                # Each line: 128 floats
                data = [np.array(row.strip().split(), dtype=np.float32) for row in f]
                signals.append(np.stack(data))
    # signals: list of 9 arrays, each shape (num_samples, 128)
    # Stack to shape (9, num_samples, 128)
    signals = np.stack(signals)  # (9, num_samples, 128)
    # Transpose to (num_samples, 128, 9)
    return signals.transpose(1, 2, 0)

def load_labels(subdir):
    file_path = os.path.join(DATASET_DIR, subdir, f'y_{subdir}.txt')
    with open(file_path, 'r') as f:
        return np.array([int(row.strip()) - 1 for row in f])  # 0-based labels

def load_subjects(subdir):
    file_path = os.path.join(DATASET_DIR, subdir, f'subject_{subdir}.txt')
    with open(file_path, 'r') as f:
        return np.array([int(row.strip()) for row in f])

def normalize_data(X):
    mean = np.mean(X, axis=(0, 1), keepdims=True)
    std = np.std(X, axis=(0, 1), keepdims=True)
    return (X - mean) / (std + 1e-8)

def preprocess_data(X, y, subjects):
    X = normalize_data(X)
    subject_data = {}
    for subj in np.unique(subjects):
        mask = (subjects == subj)
        subject_data[subj] = {
            'X': X[mask],
            'y': y[mask],
            'subject_id': subj
        }
    return subject_data

def load_and_preprocess():
    if not os.path.exists(DATASET_DIR):
        print(f"[INFO] UCI HAR dataset not found in '{DATASET_DIR}'.")
        return None

    X_train = load_signals('train')
    X_test = load_signals('test')
    X = np.concatenate((X_train, X_test), axis=0)
    y_train = load_labels('train')
    y_test = load_labels('test')
    y = np.concatenate((y_train, y_test), axis=0)
    subjects_train = load_subjects('train')
    subjects_test = load_subjects('test')
    subjects = np.concatenate((subjects_train, subjects_test), axis=0)
    subject_data = preprocess_data(X, y, subjects)

    # Split each subject's data into train/validate
    from sklearn.model_selection import train_test_split
    processed_data = {}
    for subj, data in subject_data.items():
        Xtr, Xval, ytr, yval = train_test_split(
            data['X'], data['y'], test_size=0.2, random_state=42)
        processed_data[subj] = {
            'train': {'X': Xtr, 'y': ytr},
            'validate': {'X': Xval, 'y': yval}
        }
    return processed_data
