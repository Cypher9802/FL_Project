import numpy as np
from pathlib import Path
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load configuration from YAML
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Extract hyperparameters
WINDOW_SIZE = config["window_size"]
STEP_SIZE = config["step_size"]
SMOTE_K = config["smote_k_neighbors"]
NOISE_LEVEL = config["noise_level"]
ROT_STD = config["rot_std"]

# Define project paths
ROOT     = Path(__file__).resolve().parent.parent
UCI_ROOT = ROOT / "UCI HAR Dataset"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_inertial(split: str):
    """
    Loads the 9 raw inertial signals for the given split.
    Returns:
      X: np.ndarray of shape (n_windows, window_size, 9)
      y: np.ndarray of shape (n_windows,)
      subj: np.ndarray of shape (n_windows,)
    """
    signals = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x','body_gyro_y','body_gyro_z',
        'total_acc_x','total_acc_y','total_acc_z'
    ]
    X_list = []
    for sig in signals:
        path = UCI_ROOT / split / 'Inertial Signals' / f"{sig}_{split}.txt"
        X_list.append(np.loadtxt(path))
    X = np.stack(X_list, axis=-1)  # shape: (n_windows, window_size, 9)
    y = np.loadtxt(UCI_ROOT / split / f"y_{split}.txt").astype(int) - 1
    subj = np.loadtxt(UCI_ROOT / split / f"subject_{split}.txt").astype(int)
    return X, y, subj

def normalize(X: np.ndarray) -> np.ndarray:
    """Min-max normalize each feature across all windows and timesteps."""
    X_min = X.min(axis=(0,1), keepdims=True)
    X_max = X.max(axis=(0,1), keepdims=True)
    return (X - X_min) / (X_max - X_min + 1e-8)

def augment_noise_warp(X: np.ndarray) -> np.ndarray:
    """Apply light Gaussian noise and occasional time-flip to simulate device variation."""
    X_noised = X + np.random.normal(0, NOISE_LEVEL, X.shape)
    if np.random.rand() < 0.3:
        X_noised = np.flip(X_noised, axis=1)
    return X_noised

def balance_smote(X: np.ndarray, y: np.ndarray):
    """
    Apply SMOTE-style oversampling to balance class counts.
    Returns resampled X (n_resampled, window_size, 9) and y (n_resampled,).
    """
    n, w, f = X.shape
    X_flat = X.reshape(n, w*f)
    sm = SMOTE(k_neighbors=SMOTE_K)
    X_res, y_res = sm.fit_resample(X_flat, y)
    X_res = X_res.reshape(-1, w, f)
    return X_res, y_res

def prepare_split(split: str):
    """
    For each subject in the split:
      - Load and normalize
      - Balance & augment (train only)
      - Stratified train/val split
    Returns a dict mapping subject_id -> ((X_train, y_train), (X_val, y_val))
    """
    X, y, subj = load_raw_inertial(split)
    X = normalize(X)
    clients = {}
    for s in np.unique(subj):
        idx = subj == s
        X_sub, y_sub = X[idx], y[idx]
        if split == 'train':
            X_sub, y_sub = balance_smote(X_sub, y_sub)
            X_sub = augment_noise_warp(X_sub)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sub, y_sub,
            test_size=0.2,
            stratify=y_sub,
            random_state=42
        )
        clients[int(s)] = ((X_tr, y_tr), (X_val, y_val))
    return clients

def save_processed():
    """
    Execute preprocessing for both 'train' and 'test' splits,
    saving subject-wise .npy files under data/processed/subject_##/.
    """
    for split in ["train", "test"]:
        clients = prepare_split(split)
        for subj_id, ((X_tr, y_tr), (X_val, y_val)) in clients.items():
            out_dir = PROC_DIR / f"subject_{subj_id:02d}"
            out_dir.mkdir(exist_ok=True)
            # Save train or test accordingly
            if split == "train":
                np.save(out_dir / "X_win_train.npy", X_tr)
                np.save(out_dir / "y_win_train.npy", y_tr)
            else:
                np.save(out_dir / "X_win_test.npy", X_val)
                np.save(out_dir / "y_win_test.npy", y_val)

if __name__ == "__main__":
    save_processed()
    print("Data preprocessing complete: .npy files saved in data/processed/")  
