import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_synthetic_multivariate(n_series=3, length=2000, trend=0.001, noise_std=0.1, seed=42):
    """
    Generates multivariate time series with:
       - linear trend
       - two seasonal components (daily and yearly-like)
       - some cross-correlation between series
    Returns shape (length, n_series)
    """
    rng = np.random.RandomState(seed)
    t = np.arange(length)
    data = np.zeros((length, n_series), dtype=np.float32)
    for i in range(n_series):
        # base seasonalities with different frequencies/phases
        s1 = 0.8 * np.sin(2 * np.pi * t / (24 + 2*i) + rng.randn()*0.1)
        s2 = 0.4 * np.sin(2 * np.pi * t / (168 + 5*i) + rng.randn()*0.2)  # weekly-ish
        tr = trend * t * (1 + 0.02*i)
        noise = rng.normal(scale=noise_std*(1+0.2*i), size=length)
        data[:, i] = s1 + s2 + tr + noise
    # add small coupling between series
    data[:, 1:] += 0.2 * data[:, :1]
    return data

class SeqDataset(Dataset):
    def __init__(self, data, input_len=48, horizon=1):
        # data: numpy array (time, features)
        self.data = data
        self.input_len = input_len
        self.horizon = horizon
        self.length = data.shape[0]
    def __len__(self):
        return max(0, self.length - self.input_len - self.horizon + 1)
    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_len]
        y = self.data[idx + self.input_len + self.horizon - 1]  # predict next step (multivariate)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
