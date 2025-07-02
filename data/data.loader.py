import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

# ========== CONFIGURATION ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'UCI HAR Dataset')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset file paths
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

# FIXED: More reasonable preprocessing parameters based on UCI HAR standard
WINDOW_SIZE = 64       # Reduced from 128 - more windows per subject
STEP_SIZE = 32         # Reduced from 64 - more overlap, more samples
NOISE_LEVEL = 0.01
ROTATION_ANGLE_STD = 0.05
MIN_SAMPLES = 10       # Reduced from 5 - more lenient threshold

def check_dataset_files():
    """Check if all required UCI HAR Dataset files exist."""
    required_files = list(TRAIN_FILES.values()) + list(TEST_FILES.values())
    required_files.extend([FEATURE_NAMES_FILE, ACTIVITY_LABELS_FILE])
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("Missing required dataset files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        logger.error("Please ensure the UCI HAR Dataset is properly extracted in the project root.")
        sys.exit(1)
    
    logger.info("All required dataset files found.")

def load_feature_names():
    """Load feature names from features.txt file, ensuring uniqueness."""
    try:
        features_df = pd.read_csv(FEATURE_NAMES_FILE, sep=r'\s+', header=None, names=['index', 'name'])
        raw_names = features_df['name'].tolist()
        
        # Make feature names unique by adding counters to duplicates
        seen = {}
        unique_names = []
        for name in raw_names:
            if name in seen:
                seen[name] += 1
                unique_names.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique_names.append(name)
        
        return unique_names
    except Exception as e:
        logger.warning(f"Could not load feature names: {e}. Using generic names.")
        return [f'feature_{i}' for i in range(561)]

def load_activity_labels():
    """Load activity labels from activity_labels.txt file."""
    try:
        activity_df = pd.read_csv(ACTIVITY_LABELS_FILE, sep=r'\s+', header=None, names=['id', 'activity'])
        return dict(zip(activity_df['id'], activity_df['activity']))
    except Exception as e:
        logger.warning(f"Could not load activity labels: {e}. Using generic labels.")
        return {i: f'activity_{i}' for i in range(1, 7)}

def load_uci_har_data():
    """Load UCI HAR Dataset and combine train/test data."""
    logger.info("Loading UCI HAR Dataset...")
    
    # Load feature names and activity labels
    feature_names = load_feature_names()
    activity_labels = load_activity_labels()
    
    # Load training data
    logger.info("Loading training data...")
    X_train = pd.read_csv(TRAIN_FILES['features'], sep=r'\s+', header=None)
    y_train = pd.read_csv(TRAIN_FILES['labels'], sep=r'\s+', header=None)
    subject_train = pd.read_csv(TRAIN_FILES['subjects'], sep=r'\s+', header=None)
    
    # Load test data
    logger.info("Loading test data...")
    X_test = pd.read_csv(TEST_FILES['features'], sep=r'\s+', header=None)
    y_test = pd.read_csv(TEST_FILES['labels'], sep=r'\s+', header=None)
    subject_test = pd.read_csv(TEST_FILES['subjects'], sep=r'\s+', header=None)
    
    # Set proper column names
    X_train.columns = feature_names
    X_test.columns = feature_names
    
    # Create combined DataFrames
    train_df = pd.concat([
        subject_train.rename(columns={0: 'subject'}),
        y_train.rename(columns={0: 'activity'}),
        X_train
    ], axis=1)
    
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
    for activity_id, count in train_df['activity'].value_counts().sort_index().items():
        activity_name = activity_labels.get(activity_id, f'Unknown_{activity_id}')
        logger.info(f"  {activity_name}: {count}")
    
    return train_df, test_df

def normalize_features(train_df, test_df):
    """Normalize features to [0, 1] range."""
    logger.info("Normalizing features to [0, 1] range...")
    
    # Separate features from labels and subjects
    feature_cols = [col for col in train_df.columns if col not in ['subject', 'activity']]
    
    # Fit scaler on training data only
    scaler = MinMaxScaler()
    train_norm = train_df.copy()
    test_norm = test_df.copy()
    
    train_norm[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_norm[feature_cols] = scaler.transform(test_df[feature_cols])
    
    logger.info("Feature normalization completed.")
    return train_norm, test_norm, scaler

def split_by_subjects(df):
    """Split dataset by subjects to simulate federated clients."""
    logger.info("Splitting data by subjects for federated learning simulation...")
    
    subjects_data = {}
    for subject_id, subject_data in df.groupby('subject'):
        # Remove subject column from individual client data
        client_data = subject_data.drop(columns=['subject']).reset_index(drop=True)
        subjects_data[subject_id] = client_data
        logger.info(f"Subject {subject_id}: {len(client_data)} samples")
    
    logger.info(f"Data split into {len(subjects_data)} clients (subjects).")
    return subjects_data

def augment_data(df, noise_level=NOISE_LEVEL, rotation_std=ROTATION_ANGLE_STD):
    """Augment data with noise and rotations."""
    feature_cols = [col for col in df.columns if col != 'activity']
    features = df[feature_cols].values
    labels = df['activity'].values
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, features.shape)
    features_noisy = features + noise
    
    # Apply small rotations
    rotation_factors = np.random.normal(1.0, rotation_std, features.shape)
    features_rotated = features_noisy * rotation_factors
    
    # Ensure values remain in valid range
    features_augmented = np.clip(features_rotated, 0, 1)
    
    # Create augmented DataFrame
    augmented_df = pd.DataFrame(features_augmented, columns=feature_cols)
    augmented_df['activity'] = labels
    
    return augmented_df

def extract_time_series_features(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Extract features from time series using sliding windows."""
    feature_cols = [col for col in df.columns if col != 'activity']
    features = df[feature_cols].values
    labels = df['activity'].values
    
    windowed_features = []
    windowed_labels = []
    
    # Sliding window approach
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window_data = features[start:end]
        window_labels = labels[start:end]
        
        # Extract statistical features from the window
        mean_features = np.mean(window_data, axis=0)
        std_features = np.std(window_data, axis=0)
        min_features = np.min(window_data, axis=0)
        max_features = np.max(window_data, axis=0)
        median_features = np.median(window_data, axis=0)
        
        # Combine all statistical features
        combined_features = np.concatenate([
            mean_features, std_features, min_features, max_features, median_features
        ])
        
        # Use majority vote for window label
        window_label = np.bincount(window_labels.astype(int)).argmax()
        
        windowed_features.append(combined_features)
        windowed_labels.append(window_label)
    
    return np.array(windowed_features), np.array(windowed_labels)

def prepare_federated_data(subjects_data, augment=True, extract_features=True):
    """Prepare data for federated learning with robust error handling."""
    logger.info("Preparing federated learning data...")
    federated_data = {}
    
    for subject_id, subject_df in subjects_data.items():
        logger.info(f"Processing subject {subject_id}...")
        
        try:
            # Apply data augmentation if requested
            if augment:
                augmented_df = augment_data(subject_df)
            else:
                augmented_df = subject_df.copy()
            
            # Extract features if requested
            if extract_features:
                features, labels = extract_time_series_features(augmented_df)
            else:
                feature_cols = [col for col in augmented_df.columns if col != 'activity']
                features = augmented_df[feature_cols].values
                labels = augmented_df['activity'].values
            
            n_samples = len(labels)
            unique_labels, counts = np.unique(labels, return_counts=True)
            n_classes = len(unique_labels)
            
            # Skip clients with too few samples
            if n_samples < MIN_SAMPLES:
                logger.warning(f"Subject {subject_id}: Too few samples ({n_samples}). Skipping.")
                continue
            
            # For very small datasets, use all data for training
            if n_samples <= 15:
                X_train = features
                y_train = labels
                # Create small validation set by duplicating some samples
                val_indices = np.random.choice(len(features), size=min(3, len(features)), replace=True)
                X_val = features[val_indices]
                y_val = labels[val_indices]
            else:
                # Normal train/test split for larger datasets
                test_size = min(0.3, max(0.1, n_classes / n_samples))
                
                try:
                    # Try stratified split first
                    if np.all(counts >= 2) and n_samples >= n_classes * 2:
                        X_train, X_val, y_train, y_val = train_test_split(
                            features, labels, test_size=test_size, random_state=42, stratify=labels
                        )
                    else:
                        # Fall back to random split
                        X_train, X_val, y_train, y_val = train_test_split(
                            features, labels, test_size=test_size, random_state=42, stratify=None
                        )
                except Exception as e:
                    logger.warning(f"Subject {subject_id}: Split failed ({e}). Using simple split.")
                    # Simple manual split as last resort
                    split_point = int(0.8 * n_samples)
                    X_train, X_val = features[:split_point], features[split_point:]
                    y_train, y_val = labels[:split_point], labels[split_point:]
            
            federated_data[subject_id] = {
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val,
                'num_samples': len(features)
            }
            
            logger.info(f"Subject {subject_id}: {len(X_train)} train, {len(X_val)} val samples")
            
        except Exception as e:
            logger.error(f"Subject {subject_id}: Processing failed with error: {e}")
            continue
    
    if not federated_data:
        logger.error("No clients successfully processed! Check your data and parameters.")
        return {}
    
    logger.info(f"Successfully processed {len(federated_data)} clients")
    return federated_data

def save_processed_data(federated_data, scaler, metadata):
    """Save processed data, scaler, and metadata to disk."""
    if not federated_data:
        logger.warning("No federated data to save.")
        return
    
    logger.info("Saving processed data to disk...")
    
    # Save federated data
    for subject_id, data in federated_data.items():
        subject_dir = os.path.join(OUTPUT_DIR, f'subject_{subject_id}')
        os.makedirs(subject_dir, exist_ok=True)
        
        for data_type, array in data.items():
            if isinstance(array, np.ndarray):
                np.save(os.path.join(subject_dir, f'{data_type}.npy'), array)
    
    # Save scaler
    import joblib
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Save preprocessing metadata
    metadata_path = os.path.join(OUTPUT_DIR, 'preprocessing_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")
    
    logger.info("All processed data saved successfully.")

def generate_data_summary(federated_data):
    """Generate comprehensive summary of the processed data."""
    if not federated_data:
        logger.warning("No clients with sufficient data for summary.")
        return {
            'total_clients': 0,
            'total_train_samples': 0,
            'total_val_samples': 0,
            'feature_dimension': 0,
            'preprocessing_params': {
                'window_size': WINDOW_SIZE,
                'step_size': STEP_SIZE,
                'noise_level': NOISE_LEVEL,
                'rotation_angle_std': ROTATION_ANGLE_STD
            },
            'client_statistics': {}
        }
    
    total_train_samples = sum(data['X_train'].shape[0] for data in federated_data.values())
    total_val_samples = sum(data['X_val'].shape[0] for data in federated_data.values())
    num_clients = len(federated_data)
    
    # Feature dimensions
    sample_features = next(iter(federated_data.values()))['X_train']
    feature_dim = sample_features.shape[1]
    
    # Client statistics
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
    """Main preprocessing pipeline."""
    logger.info("Starting UCI HAR Dataset preprocessing pipeline...")
    logger.info("=" * 60)
    
    try:
        # Step 1: Check dataset files
        check_dataset_files()
        
        # Step 2: Load raw data
        train_df, test_df = load_uci_har_data()
        
        # Step 3: Normalize data to [0, 1] range
        train_normalized, test_normalized, scaler = normalize_features(train_df, test_df)
        
        # Step 4: Split by subjects (simulate distributed clients)
        train_subjects = split_by_subjects(train_normalized)
        test_subjects = split_by_subjects(test_normalized)
        
        # Step 5: Prepare federated data with augmentation and feature extraction
        logger.info("Processing training data for federated learning...")
        federated_train_data = prepare_federated_data(
            train_subjects, 
            augment=True, 
            extract_features=True
        )
        
        logger.info("Processing test data for evaluation...")
        federated_test_data = prepare_federated_data(
            test_subjects, 
            augment=False,  # No augmentation for test data
            extract_features=True
        )
        
        # Step 6: Generate summary and metadata
        train_summary = generate_data_summary(federated_train_data)
        test_summary = generate_data_summary(federated_test_data)
        
        # Check if we have enough data
        if train_summary['total_clients'] == 0:
            logger.error("No training clients with sufficient data. Exiting.")
            sys.exit(1)
        
        if test_summary['total_clients'] == 0:
            logger.warning("No test clients with sufficient data. Continuing with training data only.")
        
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
        
        # Step 7: Save all processed data
        save_processed_data(federated_train_data, scaler, metadata)
        
        # Also save test data separately if available
        if federated_test_data:
            test_output_dir = os.path.join(OUTPUT_DIR, 'test')
            os.makedirs(test_output_dir, exist_ok=True)
            
            for subject_id, data in federated_test_data.items():
                subject_dir = os.path.join(test_output_dir, f'subject_{subject_id}')
                os.makedirs(subject_dir, exist_ok=True)
                
                for data_type, array in data.items():
                    if isinstance(array, np.ndarray):
                        np.save(os.path.join(subject_dir, f'{data_type}.npy'), array)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Training clients: {len(federated_train_data)}")
        logger.info(f"Test clients: {len(federated_test_data) if federated_test_data else 0}")
        logger.info(f"Feature dimension: {train_summary['feature_dimension']}")
        logger.info(f"Total training samples: {train_summary['total_train_samples']}")
        logger.info(f"Total validation samples: {train_summary['total_val_samples']}")
        logger.info(f"Processed data saved to: {OUTPUT_DIR}")
        logger.info("Ready for federated learning training!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
