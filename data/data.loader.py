import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import logging

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'UCI HAR Dataset')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILES = {
    'features': os.path.join(DATASET_DIR, 'train', 'X_train.txt'),
    'labels': os.path.join(DATASET_DIR, 'train', 'y_train.txt'),
    'subjects': os.path.join(DATASET_DIR, 'train', 'subject_train.txt')
}
TEST_FILES = {
    'features': os.path.join(DATASET_DIR, 'test', 'X_test.txt'),
    'labels': os.path.join(DATASET_DIR, 'test', 'y_test.txt'),
    'subjects': os.path.join(DATASET_DIR, 'test', 'subject_test.txt')
}
FEATURE_NAMES_FILE = os.path.join(DATASET_DIR, 'features.txt')
ACTIVITY_LABELS_FILE = os.path.join(DATASET_DIR, 'activity_labels.txt')

WINDOW_SIZE = 128
STEP_SIZE = 64
NOISE_LEVEL = 0.01
ROTATION_ANGLE_STD = 0.05
NUM_FEATURES = 561
NUM_ACTIVITIES = 6

def check_dataset_files():
    required_files = list(TRAIN_FILES.values()) + list(TEST_FILES.values())
    required_files += [FEATURE_NAMES_FILE, ACTIVITY_LABELS_FILE]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        logger.error("Missing files: %s", missing)
        sys.exit(1)
    logger.info("All required dataset files found.")

def load_feature_names():
    raw = pd.read_csv(FEATURE_NAMES_FILE, sep=r"\s+", header=None, names=["idx", "name"])["name"]
    seen, unique = {}, []
    for name in raw:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)
    return unique

def load_activity_labels():
    try:
        df = pd.read_csv(ACTIVITY_LABELS_FILE, sep=r'\s+', header=None, names=['id', 'activity'])
        return dict(zip(df['id'], df['activity']))
    except Exception:
        return {i: f'activity_{i}' for i in range(1, NUM_ACTIVITIES + 1)}

def load_uci_har_data():
    logger.info("Loading UCI HAR Dataset...")
    feature_names = load_feature_names()
    activity_labels = load_activity_labels()
    X_train = pd.read_csv(TRAIN_FILES['features'], sep=r'\s+', header=None)
    y_train = pd.read_csv(TRAIN_FILES['labels'], sep=r'\s+', header=None)
    subject_train = pd.read_csv(TRAIN_FILES['subjects'], sep=r'\s+', header=None)
    X_train.columns = feature_names
    train_df = pd.concat([
        subject_train.rename(columns={0: 'subject'}),
        y_train.rename(columns={0: 'activity'}),
        X_train
    ], axis=1)
    X_test = pd.read_csv(TEST_FILES['features'], sep=r'\s+', header=None)
    y_test = pd.read_csv(TEST_FILES['labels'], sep=r'\s+', header=None)
    subject_test = pd.read_csv(TEST_FILES['subjects'], sep=r'\s+', header=None)
    X_test.columns = feature_names
    test_df = pd.concat([
        subject_test.rename(columns={0: 'subject'}),
        y_test.rename(columns={0: 'activity'}),
        X_test
    ], axis=1)
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    logger.info(f"Number of subjects in training: {train_df['subject'].nunique()}")
    logger.info(f"Number of subjects in test: {test_df['subject'].nunique()}")
    logger.info(f"Activity distribution in training:")
    for aid, count in train_df['activity'].value_counts().sort_index().items():
        logger.info(f"  {activity_labels.get(aid, aid)}: {count}")
    return train_df, test_df

def normalize_features(train_df, test_df):
    feature_cols = [col for col in train_df.columns if col not in ['subject', 'activity']]
    scaler = MinMaxScaler()
    train_norm = train_df.copy()
    test_norm = test_df.copy()
    train_norm[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_norm[feature_cols] = scaler.transform(test_df[feature_cols])
    logger.info("Feature normalization completed.")
    return train_norm, test_norm, scaler

def split_by_subjects(df):
    logger.info("Splitting data by subjects for federated learning simulation...")
    subjects_data = {}
    for subject_id, subject_data in df.groupby('subject'):
        client_data = subject_data.drop(columns=['subject']).reset_index(drop=True)
        subjects_data[subject_id] = client_data
        logger.info(f"Subject {subject_id}: {len(client_data)} samples")
    logger.info(f"Data split into {len(subjects_data)} clients (subjects).")
    return subjects_data

def augment_data(df, noise_level=NOISE_LEVEL, rotation_std=ROTATION_ANGLE_STD):
    feature_cols = [col for col in df.columns if col != 'activity']
    features = df[feature_cols].values
    labels = df['activity'].values
    noise = np.random.normal(0, noise_level, features.shape)
    features_noisy = features + noise
    rotation_factors = np.random.normal(1.0, rotation_std, features.shape)
    features_rotated = features_noisy * rotation_factors
    features_augmented = np.clip(features_rotated, 0, 1)
    augmented_df = pd.DataFrame(features_augmented, columns=feature_cols)
    augmented_df['activity'] = labels
    return augmented_df

def extract_time_series_features(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    feature_cols = [col for col in df.columns if col != 'activity']
    features = df[feature_cols].values
    labels = df['activity'].values
    windowed_features, windowed_labels = [], []
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window_data = features[start:end]
        window_labels = labels[start:end]
        mean_features = np.mean(window_data, axis=0)
        std_features = np.std(window_data, axis=0)
        min_features = np.min(window_data, axis=0)
        max_features = np.max(window_data, axis=0)
        median_features = np.median(window_data, axis=0)
        combined_features = np.concatenate([
            mean_features, std_features, min_features, max_features, median_features
        ])
        window_label = np.bincount(window_labels.astype(int)).argmax()
        windowed_features.append(combined_features)
        windowed_labels.append(window_label)
    return np.array(windowed_features), np.array(windowed_labels)

def prepare_federated_data(subjects_data, augment=True, extract_features=True, min_samples=5):
    logger.info("Preparing federated learning data...")
    federated_data = {}
    for subject_id, subject_df in subjects_data.items():
        logger.info(f"Processing subject {subject_id}...")
        if augment:
            augmented_df = augment_data(subject_df)
        else:
            augmented_df = subject_df.copy()
        if extract_features:
            features, labels = extract_time_series_features(augmented_df)
        else:
            feature_cols = [col for col in augmented_df.columns if col != 'activity']
            features = augmented_df[feature_cols].values
            labels = augmented_df['activity'].values

        n_samples = len(labels)
        n_classes = len(np.unique(labels))

        if n_samples < min_samples or n_samples < n_classes:
            logger.warning(
                f"Subject {subject_id}: Too few samples after feature extraction "
                f"({n_samples} samples, {n_classes} classes). Skipping this client."
            )
            continue

        unique, counts = np.unique(labels, return_counts=True)
        if np.all(counts >= 2) and n_samples >= n_classes * 2:
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    features, labels, test_size=0.2, random_state=42, stratify=labels
                )
            except ValueError as e:
                logger.warning(f"Subject {subject_id}: Stratified split failed ({e}). Using random split.")
                X_train, X_val, y_train, y_val = train_test_split(
                    features, labels, test_size=0.2, random_state=42, stratify=None
                )
        else:
            if n_samples > 1:
                logger.warning(
                    f"Subject {subject_id}: Not all classes have >=2 samples or too few samples for stratification. Using random split."
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    features, labels, test_size=0.2, random_state=42, stratify=None
                )
            else:
                logger.warning(f"Subject {subject_id}: Only one sample. Skipping this client.")
                continue

        federated_data[subject_id] = {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'num_samples': len(features)
        }
        logger.info(f"Subject {subject_id}: {len(X_train)} train, {len(X_val)} val samples")
    return federated_data

def save_processed_data(federated_data, scaler, metadata):
    logger.info("Saving processed data to disk...")
    for subject_id, data in federated_data.items():
        subject_dir = os.path.join(OUTPUT_DIR, f'subject_{subject_id}')
        os.makedirs(subject_dir, exist_ok=True)
        for data_type, array in data.items():
            if isinstance(array, np.ndarray):
                np.save(os.path.join(subject_dir, f'{data_type}.npy'), array)
    import joblib
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
    with open(os.path.join(OUTPUT_DIR, 'preprocessing_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("All processed data saved successfully.")

def generate_data_summary(federated_data):
    total_train_samples = sum(data['X_train'].shape[0] for data in federated_data.values())
    total_val_samples = sum(data['X_val'].shape[0] for data in federated_data.values())
    num_clients = len(federated_data)
    sample_features = next(iter(federated_data.values()))['X_train']
    feature_dim = sample_features.shape[1]
    client_stats = {}
    for subject_id, data in federated_data.items():
        client_stats[subject_id] = {
            'train_samples': int(data['X_train'].shape[0]),
            'val_samples': int(data['X_val'].shape[0]),
            'total_samples': int(data['num_samples'])
        }
    summary = {
        'total_clients': num_clients,
        'total_train_samples': total_train_samples,
        'total_val_samples': total_val_samples,
        'feature_dimension': feature_dim,
        'preprocessing_params': {
            'window_size': WINDOW_SIZE,
            'step_size': STEP_SIZE,
            'noise_level': NOISE_LEVEL,
            'rotation_angle_std': ROTATION_ANGLE_STD
        },
        'client_statistics': client_stats
    }
    logger.info(f"Data summary: {num_clients} clients, {total_train_samples} train samples, {feature_dim} features")
    return summary

def main():
    logger.info("Starting UCI HAR Dataset preprocessing pipeline...")
    logger.info("=" * 60)
    check_dataset_files()
    train_df, test_df = load_uci_har_data()
    train_normalized, test_normalized, scaler = normalize_features(train_df, test_df)
    train_subjects = split_by_subjects(train_normalized)
    test_subjects = split_by_subjects(test_normalized)
    logger.info("Processing training data for federated learning...")
    federated_train_data = prepare_federated_data(train_subjects, augment=True, extract_features=True)
    logger.info("Processing test data for evaluation...")
    federated_test_data = prepare_federated_data(test_subjects, augment=False, extract_features=True)
    train_summary = generate_data_summary(federated_train_data)
    test_summary = generate_data_summary(federated_test_data)
    metadata = {
        'dataset': 'UCI HAR Dataset',
        'preprocessing_date': pd.Timestamp.now().isoformat(),
        'train_summary': train_summary,
        'test_summary': test_summary,
        'normalization': 'MinMaxScaler [0, 1]',
        'augmentation': 'Gaussian noise + rotation simulation',
        'feature_extraction': 'Sliding window with statistical features',
        'federated_setup': 'One subject per client'
    }
    save_processed_data(federated_train_data, scaler, metadata)
    test_output_dir = os.path.join(OUTPUT_DIR, 'test')
    os.makedirs(test_output_dir, exist_ok=True)
    for subject_id, data in federated_test_data.items():
        subject_dir = os.path.join(test_output_dir, f'subject_{subject_id}')
        os.makedirs(subject_dir, exist_ok=True)
        for data_type, array in data.items():
            if isinstance(array, np.ndarray):
                np.save(os.path.join(subject_dir, f'{data_type}.npy'), array)
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Training clients: {len(federated_train_data)}")
    logger.info(f"Test clients: {len(federated_test_data)}")
    logger.info(f"Feature dimension: {train_summary['feature_dimension']}")
    logger.info(f"Total training samples: {train_summary['total_train_samples']}")
    logger.info(f"Total validation samples: {train_summary['total_val_samples']}")
    logger.info(f"Processed data saved to: {OUTPUT_DIR}")
    logger.info("Ready for federated learning training!")

if __name__ == "__main__":
    main()
