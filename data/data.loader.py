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
    'labels':   os.path.join(DATASET_DIR, 'train', 'y_train.txt'),
    'subjects': os.path.join(DATASET_DIR, 'train', 'subject_train.txt')
}
TEST_FILES = {
    'features': os.path.join(DATASET_DIR, 'test',  'X_test.txt'),
    'labels':   os.path.join(DATASET_DIR, 'test',  'y_test.txt'),
    'subjects': os.path.join(DATASET_DIR, 'test',  'subject_test.txt')
}
FEATURE_NAMES_FILE     = os.path.join(DATASET_DIR, 'features.txt')
ACTIVITY_LABELS_FILE   = os.path.join(DATASET_DIR, 'activity_labels.txt')

# Use the original windows—no secondary re-windowing
MIN_SAMPLES = 1  # allow all subjects
WINDOW_SIZE = None
STEP_SIZE = None

def check_dataset_files():
    required = list(TRAIN_FILES.values()) + list(TEST_FILES.values()) + [FEATURE_NAMES_FILE, ACTIVITY_LABELS_FILE]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        logger.error("Missing files: %s", missing)
        sys.exit(1)
    logger.info("All required dataset files found.")

def load_feature_names():
    raw = pd.read_csv(FEATURE_NAMES_FILE, sep=r'\s+', header=None, names=['idx','name'])['name']
    seen, uniq = {}, []
    for n in raw:
        seen[n] = seen.get(n,0) + 1
        uniq.append(f"{n}_{seen[n]-1}" if seen[n]>1 else n)
    return uniq

def load_activity_labels():
    df = pd.read_csv(ACTIVITY_LABELS_FILE, sep=r'\s+', header=None, names=['id','act'])
    return dict(zip(df['id'], df['act']))

def load_uci_har_data():
    fnames = load_feature_names()
    acts   = load_activity_labels()
    # Train
    Xtr = pd.read_csv(TRAIN_FILES['features'], sep=r'\s+', header=None)
    ytr = pd.read_csv(TRAIN_FILES['labels'], sep=r'\s+', header=None)
    strn = pd.read_csv(TRAIN_FILES['subjects'], sep=r'\s+', header=None)
    Xtr.columns = fnames
    dftr = pd.concat([strn.rename(columns={0:'subject'}), ytr.rename(columns={0:'activity'}), Xtr], axis=1)
    # Test
    Xte = pd.read_csv(TEST_FILES['features'], sep=r'\s+', header=None)
    yte = pd.read_csv(TEST_FILES['labels'], sep=r'\s+', header=None)
    stst= pd.read_csv(TEST_FILES['subjects'], sep=r'\s+', header=None)
    Xte.columns = fnames
    dfte = pd.concat([stst.rename(columns={0:'subject'}), yte.rename(columns={0:'activity'}), Xte], axis=1)
    logger.info(f"Loaded train {dftr.shape}, test {dfte.shape}")
    return dftr, dfte

def normalize(df_train, df_test):
    feats = [c for c in df_train.columns if c not in ('subject','activity')]
    scaler = MinMaxScaler()
    df_train[feats] = scaler.fit_transform(df_train[feats])
    df_test[feats]  = scaler.transform(df_test[feats])
    return df_train, df_test, scaler

def split_by_subject(df):
    out = {}
    for sid, grp in df.groupby('subject'):
        out[sid] = grp.drop(columns='subject').reset_index(drop=True)
    return out

def augment(df, noise_level=0.01, rot_std=0.05):
    cols = [c for c in df.columns if c!='activity']
    X = df[cols].values
    y = df['activity'].values
    noise = np.random.normal(0, noise_level, X.shape)
    Xn = np.clip(X + noise, 0, 1)
    rf = np.random.normal(1.0, rot_std, X.shape)
    Xr = np.clip(Xn * rf, 0, 1)
    df2= pd.DataFrame(Xr, columns=cols); df2['activity']=y
    return df2

def prepare_fed(subjects, augment_data=True):
    fed = {}
    for sid, df in subjects.items():
        logger.info(f"Subject {sid}: {len(df)} samples")
        if augment_data:
            df = augment(df)
        X = df.drop(columns='activity').values
        y = df['activity'].values
        if len(y)<MIN_SAMPLES:
            logger.warning(f"Subject {sid}: too few samples, skipping")
            continue
        # stratify if possible
        try:
            Xtr, Xvl, ytr, yvl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)
        except:
            Xtr, Xvl, ytr, yvl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
        fed[sid] = {'X_train':Xtr, 'X_val':Xvl, 'y_train':ytr, 'y_val':yvl}
    return fed

def save_fed(fed, scaler):
    for sid,data in fed.items():
        ddir = os.path.join(OUTPUT_DIR, f'subject_{sid}')
        os.makedirs(ddir, exist_ok=True)
        for k,v in data.items():
            np.save(os.path.join(ddir, f'{k}.npy'), v)
    import joblib
    joblib.dump(scaler, os.path.join(OUTPUT_DIR,'scaler.joblib'))

def main():
    check_dataset_files()
    dftr, dfte = load_uci_har_data()
    dftr, dfte, scaler = normalize(dftr, dfte)
    tr_subs = split_by_subject(dftr)
    te_subs = split_by_subject(dfte)
    fed_train = prepare_fed(tr_subs, augment_data=True)
    fed_test  = prepare_fed(te_subs, augment_data=False)
    if not fed_train:
        logger.error("No training clients processed.")
        sys.exit(1)
    save_fed(fed_train, scaler)
    if fed_test:
        save_fed(fed_test, scaler)
    logger.info(f"Processed {len(fed_train)} train clients, {len(fed_test)} test clients")

if __name__=="__main__":
    main()
