import numpy as np
import torch
from torch.utils.data import Dataset
import os

def normalize(x, mode="zscore"):
    x = x.astype(np.float32)
    if mode == "zscore":
        mu = x.mean(axis=(1,2), keepdims=True)
        std = x.std(axis=(1,2), keepdims=True) + 1e-6
        return (x - mu)/std
    elif mode == "max":
        mx = np.max(np.abs(x), axis=(1,2), keepdims=True) + 1e-6
        return x/mx
    return x

def extract_patches(arr, patch):
    C,H,W = arr.shape
    pad = patch//2
    arrp = np.pad(arr, ((0,0),(pad,pad),(pad,pad)), mode="reflect")
    s0,s1,s2 = arrp.strides
    shape = (H, W, C, patch, patch)
    strides = (s1, s2, s0, s1, s2)
    patches = np.lib.stride_tricks.as_strided(arrp, shape=shape, strides=strides, writeable=False)
    return patches.reshape(H*W, C, patch, patch)

class HSICDPatchDataset(Dataset):
    def __init__(self, root, t1, t2, label, mask=None, patch=5, split="train",
                 train_ratio=0.8, max_train_samples=20000, balance=True, normalize_mode="zscore", seed=42):
        self.root = root
        T1 = np.load(os.path.join(root, t1))
        T2 = np.load(os.path.join(root, t2))
        Y = np.load(os.path.join(root, label))
        assert T1.shape == T2.shape
        C,H,W = T1.shape
        if mask is not None and os.path.exists(os.path.join(root, mask)):
            M = np.load(os.path.join(root, mask)).astype(bool)
        else:
            M = np.ones((H,W), dtype=bool)
        T1 = normalize(T1, normalize_mode)
        T2 = normalize(T2, normalize_mode)

        self.p1 = extract_patches(T1, patch)
        self.p2 = extract_patches(T2, patch)
        self.labels = Y.reshape(-1).astype(np.int64)
        self.valid = M.reshape(-1)

        idx = np.where(self.valid)[0].tolist()
        idx = [i for i in idx if self.labels[i] in (0,1)]
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)

        ntrain = int(len(idx)*train_ratio)
        train_idx = set(idx[:ntrain]); test_idx = set(idx[ntrain:])

        if split == "train":
            pool = [i for i in idx if i in train_idx]
            if balance:
                pos = [i for i in pool if self.labels[i] == 1]
                neg = [i for i in pool if self.labels[i] == 0]
                m = min(len(pos), len(neg), max_train_samples//2)
                rng.shuffle(pos); rng.shuffle(neg)
                self.indices = pos[:m] + neg[:m]
            else:
                self.indices = pool[:max_train_samples]
        else:
            self.indices = [i for i in idx if i in test_idx]

        rng.shuffle(self.indices)
        self.C = C; self.patch = patch
        self.H = H; self.W = W

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        x1 = torch.from_numpy(self.p1[j])
        x2 = torch.from_numpy(self.p2[j])
        y = int(self.labels[j])
        return x1, x2, torch.tensor(y, dtype=torch.long)
