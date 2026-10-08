"""
Dataset for NewHandPD: paired handwriting image + 6-channel sensor signal.
Preprocessing strictly matches MfamaNet official inference.py.
"""
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

# Official image transform from inference.py
IMAGE_TRANSFORM = T.Compose([
    T.Resize([224, 224], antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TARGET_LEN = 10000
TRIM_RATIO = 0.05


def preprocess_signal(signal_path):
    """Match official preprocess_signal in inference.py.
    Returns tensor [TARGET_LEN, 6].
    """
    sig = torch.as_tensor(
        pd.read_csv(signal_path).values, dtype=torch.float32)  # [L, 6]

    L = sig.shape[0]
    if TRIM_RATIO > 0 and L > 50:
        trim_len = int(L * TRIM_RATIO)
        if L - 2 * trim_len > 20:
            sig = sig[trim_len: L - trim_len]

    # Per-channel Z-score
    mean = sig.mean(dim=0, keepdim=True)
    std = sig.std(dim=0, keepdim=True).clamp(min=1e-8)
    sig = (sig - mean) / std

    # Interpolate to fixed length
    sig = sig.T.unsqueeze(0)                              # [1, C, L]
    sig = F.interpolate(sig, size=TARGET_LEN, mode='linear',
                        align_corners=True)               # [1, C, TARGET_LEN]
    sig = sig.squeeze(0).T                                # [TARGET_LEN, C]
    return sig


class ImageSignalDataset(Dataset):
    def __init__(self, index_csv, root='.', preload=False):
        self.root = root
        df = pd.read_csv(index_csv, header=None, names=['img', 'sig', 'label'])
        self.images = df['img'].tolist()
        self.signals = df['sig'].tolist()
        self.labels = df['label'].astype(int).tolist()
        self._cache = {}
        if preload:
            for i in range(len(self)):
                self._cache[i] = self._load(i)

    def _load(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        sig_path = os.path.join(self.root, self.signals[idx])
        img = Image.open(img_path).convert('RGB')
        img = IMAGE_TRANSFORM(img)                       # [3, 224, 224]
        sig = preprocess_signal(sig_path)                # [10000, 6]
        return img, sig, self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        return self._load(idx)


def get_dataloaders(index_csv, batch_size=8, test_size=0.2, root='.'):
    """Return (train_loader, test_loader) with stratified split."""
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(index_csv, header=None, names=['img', 'sig', 'label'])

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['label']
    )

    tmp = 'tmp_train.csv'
    tmp2 = 'tmp_test.csv'
    train_df.to_csv(tmp, index=False, header=False)
    test_df.to_csv(tmp2, index=False, header=False)

    train_ds = ImageSignalDataset(tmp, root=root, preload=True)
    test_ds = ImageSignalDataset(tmp2, root=root, preload=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, drop_last=False)
    return train_loader, test_loader
