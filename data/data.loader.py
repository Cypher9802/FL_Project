import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Suppress only known, harmless warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

# ========== CONFIGURATION ==========
DATA_DIR = "data/"
ACCEL_FILENAME = "accel.csv"   # Update if your filename differs
GYRO_FILENAME = "gyro.csv"     # Update if your filename differs
SUBJECT_COL = "subject"
LABEL_COL = "activity"

UCI_HAR_LINK = "https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones"

# ========== CHECK DATASET PRESENCE ==========
accel_path = os.path.join(DATA_DIR, ACCEL_FILENAME)
gyro_path = os.path.join(DATA_DIR, GYRO_FILENAME)

if not (os.path.exists(accel_path) and os.path.exists(gyro_path)):
    print("\n[ERROR] Required dataset files not found.")
    print(f"Expected: {accel_path} and {gyro_path}")
    print("Please download and extract the UCI HAR Dataset from the official source:")
    print(UCI_HAR_LINK)
    print("After downloading, place the relevant CSV files in the 'data/' directory.")
    sys.exit(1)

# ========== DATA LOADING ==========
def load_data(accel_path, gyro_path):
    """Load accelerometer and gyroscope CSVs as DataFrames."""
    accel = pd.read_csv(accel_path)
    gyro = pd.read_csv(gyro_path)
    return accel, gyro

# ========== PREPROCESSING ==========
def normalize_data(df):
    """Normalize each feature to [0, 1] and fill NaNs with column median."""
    scaler = MinMaxScaler()
    df_filled = df.fillna(df.median(numeric_only=True))
    normed = scaler.fit_transform(df_filled)
    return pd.DataFrame(normed, columns=df.columns)

def split_by_subject(df, subject_col=SUBJECT_COL):
    """Split DataFrame into a dict of subject_id: DataFrame."""
    return {subject: group.drop(columns=[subject_col])
            for subject, group in df.groupby(subject_col)}

def augment_data(df, noise_level=0.01):
    """Add small Gaussian noise to simulate sensor variation."""
    noisy = df + np.random.normal(0, noise_level, df.shape)
    return noisy

def extract_features(df, window_size=128, step=64):
    """Extract mean and std features from sliding windows."""
    features = []
    for start in range(0, len(df) - window_size + 1, step):
        window = df.iloc[start:start+window_size]
        feats = np.concatenate([window.mean().values, window.std().values])
        features.append(feats)
    return np.array(features)

def extract_labels(df, window_size=128, step=64, label_col=LABEL_COL):
    """Extract majority label for each window."""
    labels = []
    for start in range(0, len(df) - window_size + 1, step):
        window = df.iloc[start:start+window_size]
        # Majority vote for label in the window
        label = window[label_col].mode()[0]
        labels.append(label)
    return np.array(labels)

# ========== MAIN PIPELINE ==========
if __name__ == "__main__":
    # Load data
    accel, gyro = load_data(accel_path, gyro_path)

    # Normalize (drop non-numeric columns for normalization, keep labels/subject separately)
    accel_features = accel.drop([SUBJECT_COL, LABEL_COL], axis=1)
    accel_norm = normalize_data(accel_features)
    accel_norm[SUBJECT_COL] = accel[SUBJECT_COL].values
    accel_norm[LABEL_COL] = accel[LABEL_COL].values

    # Split by subject (simulate federated clients)
    subjects_data = split_by_subject(accel_norm, subject_col=SUBJECT_COL)

    # For each subject/client: augment, extract features, extract labels
    client_features = {}
    client_labels = {}
    for subj, df in subjects_data.items():
        # Separate features and labels
        features_only = df.drop([LABEL_COL], axis=1)
        labels_only = df[[LABEL_COL]]

        # Augment data
        augmented = augment_data(features_only)

        # Reattach labels for windowing
        augmented_df = pd.DataFrame(augmented, columns=features_only.columns)
        augmented_df[LABEL_COL] = labels_only.values

        # Feature extraction
        feats = extract_features(augmented_df.drop([LABEL_COL], axis=1))
        labs = extract_labels(augmented_df, label_col=LABEL_COL)

        client_features[subj] = feats
        client_labels[subj] = labs

    # Example: Train/test split for one client (for local training simulation)
    for subj in client_features:
        X_train, X_test, y_train, y_test = train_test_split(
            client_features[subj], client_labels[subj], test_size=0.2, random_state=42, shuffle=True)
        print(f"Client {subj}: Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    print("Data loading and preprocessing complete. Ready for federated training.")

