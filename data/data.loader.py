import os, random, numpy as np, pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R

class UCIHARDataset(Dataset):
    def __init__(self, features, labels, augment_config=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augment = augment_config or {}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].clone()
        y = self.labels[idx]
        if self.augment.get('enable_augmentation', False):
            x = self._augment(x.numpy())
            x = torch.FloatTensor(x)
        return x, y

    def _augment(self, arr):
        if random.random() < float(self.augment.get('augmentation_probability', 0.5)):
            # noise
            std = float(self.augment.get('noise_std', 0.05))
            arr += np.random.normal(0, std, arr.shape)
            # rotation on first 9 dims
            angle = np.radians(float(self.augment.get('rotation_angle', 15)))
            if arr.size >= 9:
                mat = arr[:9].reshape(3,3)
                axis = np.random.randn(3); axis /= np.linalg.norm(axis)
                rot = R.from_rotvec(angle * axis)
                mat = rot.apply(mat)
                arr[:9] = mat.flatten()
            # scaling
            lo, hi = map(float, self.augment.get('scaling_factor', [0.9,1.1]))
            arr *= random.uniform(lo, hi)
            # time shift
            shift = random.randint(-int(self.augment.get('time_shift',5)),
                                    int(self.augment.get('time_shift',5)))
            if shift>0:
                arr = np.concatenate([np.zeros(shift), arr[:-shift]])
            elif shift<0:
                arr = np.concatenate([arr[-shift:], np.zeros(-shift)])
        return arr

class UCIHARDataLoader:
    def __init__(self, config):
        self.cfg = config
        self.scaler = StandardScaler()

    def load_data(self):
        d = "UCI HAR Dataset"
        if not os.path.exists(d):
            raise FileNotFoundError(f"Dataset folder '{d}' not found.")
        train_X = pd.read_csv(os.path.join(d,"train","X_train.txt"), sep='\s+', header=None).values
        test_X  = pd.read_csv(os.path.join(d,"test","X_test.txt"),   sep='\s+', header=None).values
        train_y = pd.read_csv(os.path.join(d,"train","y_train.txt"), sep='\s+', header=None).values.ravel()
        test_y  = pd.read_csv(os.path.join(d,"test","y_test.txt"),   sep='\s+', header=None).values.ravel()
        train_s = pd.read_csv(os.path.join(d,"train","subject_train.txt"), sep='\s+', header=None).values.ravel()
        test_s  = pd.read_csv(os.path.join(d,"test","subject_test.txt"),   sep='\s+', header=None).values.ravel()
        X = np.vstack([train_X, test_X])
        y = np.hstack([train_y, test_y]) - 1
        s = np.hstack([train_s, test_s])
        return X, y, s

    def get_data_loaders(self):
        X, y, s = self.load_data()
        if self.cfg['dataset']['normalize']:
            X = self.scaler.fit_transform(X)
        clients = {}
        for sid in np.unique(s):
            mask = s==sid
            clients[int(sid-1)] = (X[mask], y[mask])
        train_loaders, test_X, test_y = {}, [], []
        for cid, (cx, cy) in clients.items():
            n = len(cx); nt = max(1, int(n*self.cfg['dataset']['test_split']))
            trX, trY = cx[nt:], cy[nt:]
            vaX, vaY = cx[:nt], cy[:nt]
            ds = UCIHARDataset(trX, trY, self.cfg['dataset']['augmentation'])
            ds.augment['enable_augmentation'] = self.cfg['dataset']['enable_augmentation']
            train_loaders[cid] = DataLoader(ds, batch_size=int(self.cfg['federated']['batch_size']), shuffle=True)
            test_X.append(vaX); test_y.append(vaY)
        vX, vY = np.vstack(test_X), np.hstack(test_y)
        vds = UCIHARDataset(vX, vY)
        val_loader = DataLoader(vds, batch_size=64, shuffle=False)
        return train_loaders, val_loader
